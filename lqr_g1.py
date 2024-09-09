import mujoco
import mujoco.viewer
import time
import numpy as np
import scipy
np.set_printoptions(suppress=True, precision=6)

BALANCE_COST        = 1000  # Balancing.
BALANCE_JOINT_COST  = 0     # Joints required for balancing.
OTHER_JOINT_COST    = .0    # Other joints.

DURATION = 12         # seconds
FRAMERATE = 60        # Hz
TOTAL_ROTATION = 15   # degrees
CTRL_RATE = 0.8       # seconds
BALANCE_STD = 0.01    # actuator units
OTHER_STD = 0.08      # actuator units


class HumYoga():

  def __init__(self):
    self.model = mujoco.MjModel.from_xml_path('./g1_description/g1.xml')
    self.data = mujoco.MjData(self.model)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True


  def get_feedforward(self):
    self.data.qacc = 0
    self.data.qpos[2] += -0.04509
    self.qpos_init = self.data.qpos.copy()
    mujoco.mj_inverse(self.model, self.data)
    force_ff = self.data.qfrc_inverse.copy()

    self.ctrl_ff = np.atleast_2d(force_ff) @ np.linalg.pinv(self.data.actuator_moment)
    self.ctrl_ff = self.ctrl_ff.flatten()  
    return self.qpos_init, self.ctrl_ff   
  
  def get_R(self): 
    self.R = np.eye(self.model.nu) 
    return self.R


  def get_Q(self):
    v = np.array([0, 0, 0, 600, 600, 600,
                  100, 100, 100, 100, 100, 100, 10000, 1000,  1000, 1000, 0, 0,
                  1000,
                  10, 10, 10, 10, 10, 100, 10, 10, 10, 10,

                  0, 0, 0, 100, 100, 100,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
    self.Q = np.diag(v)

    return self.Q
  

  
  def get_dynamics_model(self):
    mujoco.mj_resetData(self.model, self.data)
    self.data.ctrl = self.ctrl_ff
    self.data.qpos = self.qpos_init

    A = np.zeros((2*self.model.nv, 2*self.model.nv))
    B = np.zeros((2*self.model.nv, self.model.nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B, None, None)
    return A, B
  
  def get_K(self):

    P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

    K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
    return K

  def main_thread(self):
    with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

      mujoco.mj_resetDataKeyframe(self.model, self.data, 1)

      self.qpos_init, self.ctrl_ff = self.get_feedforward()

      self.R = self.get_R()
      self.Q = self.get_Q()
      
      self.A, self.B = self.get_dynamics_model()

      K = self.get_K()

      # reset for running
      mujoco.mj_resetData(self.model, self.data)
      self.data.qpos = self.qpos_init
      self.data.ctrl = self.ctrl_ff
      self.dq = np.zeros(self.model.nv)

      while viewer.is_running():

        mujoco.mj_differentiatePos(self.model, self.dq, 1, self.qpos_init, self.data.qpos)
        dx = np.hstack((self.dq, self.data.qvel)).T

        self.data.ctrl = self.ctrl_ff - K @ dx

        mujoco.mj_step(self.model, self.data)
        

        with viewer.lock():
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

        viewer.sync()
        
        # time.sleep(10)



if __name__ == '__main__':
  HumYoga().main_thread()

