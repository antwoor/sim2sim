# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of a Laikago robot."""

import os
import inspect
import copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import enum
import math
import re
import numpy as np
from numpy.typing import NDArray
import pybullet as pyb  # pytype: disable=import-error
import time

from motion_imitation.robots import a1_robot_velocity_estimator
from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config
from scipy.spatial.transform import Rotation 

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi
TWO_PI = 2 * math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
MAX_JOINT_VELOCITY = np.inf  # rad/s (was 11)
MAX_TORQUE = 42

_DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.13, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.13, 0),
)

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0.0, 0.9, -1.8,  # FR leg
                             0.0, 0.9, -1.8,     # FL leg
                             0.0, 0.9, -1.8,    # RR leg
                             0.0, 0.9, -1.8])   # RL leg

MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]

# Update these constants to match XML joint limits
MOTOR_MINS = np.array([
    -0.802851,  # hip abduction (from XML: range="-0.802851 0.802851")
    -1.0472,    # hip rotation (from XML: range="-1.0472 4.18879")
    -2.69653    # knee (from XML: range="-2.69653 -0.916298")
] * 4)

MOTOR_MAXS = np.array([
    0.802851,   # hip abduction
    4.18879,    # hip rotation
    -0.916298   # knee
] * 4)

MOTOR_OFFSETS = np.array([
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])

MOTOR_USED = np.array([
    [0.01, 0.99],
    [0.01, 0.90],
    [0.01, 0.60],
] * 4)

STANDING_POSE = np.array([0, -0.2, 1.0] * 4)


def unnormalize_action(action, clip=True):
  if clip:
    action = np.clip(action, -1, 1)
  action = action / 2 + 0.5
  lo = MOTOR_MINS * (1 - MOTOR_USED[:, 0]) + MOTOR_MAXS * MOTOR_USED[:, 0]
  hi = MOTOR_MINS * (1 - MOTOR_USED[:, 1]) + MOTOR_MAXS * MOTOR_USED[:, 1]
  action = action * (hi - lo) + lo
  action += MOTOR_OFFSETS
  return action

def normalize_action(action, clip=True):
  action -= MOTOR_OFFSETS
  lo = MOTOR_MINS * (1 - MOTOR_USED[:, 0]) + MOTOR_MAXS * MOTOR_USED[:, 0]
  hi = MOTOR_MINS * (1 - MOTOR_USED[:, 1]) + MOTOR_MAXS * MOTOR_USED[:, 1]
  action = (action - lo) / (hi - lo)
  action = (action - 0.5) * 2
  if clip:
    action = np.clip(action, -1, 1)
  return action



# INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)
# print(normalize_action(INIT_MOTOR_ANGLES))
# print(STANDING_POSE)
# import sys; sys.exit()


HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = os.path.join(parentdir, "motion_imitation/utilities/a1/a1_original.urdf")

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

# Empirical values from real a1_original.
ACCELEROMETER_VARIANCE = 0.03059
JOINT_VELOCITY_VARIANCE = 0.006206


class VelocitySource(enum.Enum):
  PYBULLET = 0
  IMU_FOOT_CONTACT = 1


# Found that these numba.jit decorators slow down the timestep from 1ms without
# to 5ms with decorators.
# @numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * l_hip_sign
  x, y, z = foot_position[0], foot_position[1], foot_position[2]
  theta_knee = -np.arccos(
      (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
      (2 * l_low * l_up))
  l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
  theta_hip = np.arcsin(-x / l) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)
  return np.array([theta_ab, theta_hip, theta_knee])


# @numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame(angles, l_hip_sign=1):
  theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * l_hip_sign
  leg_distance = np.sqrt(l_up**2 + l_low**2 +
                         2 * l_up * l_low * np.cos(theta_knee))
  eff_swing = theta_hip + theta_knee / 2

  off_x_hip = -leg_distance * np.sin(eff_swing)
  off_z_hip = -leg_distance * np.cos(eff_swing)
  off_y_hip = l_hip

  off_x = off_x_hip
  off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  return np.array([off_x, off_y, off_z])


# @numba.jit(nopython=True, cache=True)
def analytical_leg_jacobian(leg_angles, leg_id):
  """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a left (1) or right(-1) leg.
  """
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * (-1)**(leg_id + 1)

  t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
  l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
  t_eff = t2 + t3 / 2
  J = np.zeros((3, 3))
  J[0, 0] = 0
  J[0, 1] = -l_eff * np.cos(t_eff)
  J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
      t_eff) / 2
  J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
  J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
  J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
      t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
  J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
  J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
  J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
      t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
  return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


# @numba.jit(nopython=True, cache=True, parallel=True)
def foot_positions_in_base_frame(foot_angles):
  foot_angles = foot_angles.reshape((4, 3))
  foot_positions = np.zeros((4, 3))
  for i in range(4):
    foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                   l_hip_sign=(-1)**(i + 1))
  return foot_positions + HIP_OFFSETS

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue, Empty
from collections import deque
import os
import inspect
import concurrent.futures as futures
import numpy as np
from scipy.spatial.transform import Rotation as scR

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from motion_imitation.robots import a1_original 
XML_PATH = os.path.join(parentdir , "sim/robots/a1/scene.xml")
_UNIT_QUATERNION = (1, 0, 0, 0)
ABDUCTION_D_GAIN = a1_original.ABDUCTION_D_GAIN
class custom_pyb():
    def invertTransform(position, orientation): #TODO посмотреть какая запись кватерниона в mujoco и в pybullet
        """
        Инвертирует трансформацию (позицию и ориентацию) в стиле MuJoCo.

        :param position: (x, y, z) – исходная позиция
        :param orientation: (qw, qx, qy, qz) – исходный кватернион (MuJoCo формат)
        :return: (inv_position, inv_orientation) – инвертированные трансформация
        """
        # Инвертируем кватернион (q⁻¹)
        orientation_xyzw = np.roll(orientation, -1)  # Переключаем (qw, qx, qy, qz) → (qx, qy, qz, qw)
        inv_quat = scR.from_quat(orientation_xyzw).inv()

        # Инвертируем позицию
        neg_position = -np.array(position)
        inv_position = inv_quat.apply(neg_position)

        return inv_position.tolist(), np.roll(inv_quat.as_quat(), 1).tolist()
    def loadURDF(*args, **kwargs):
        pass
    def getNumJoints(*args, **kwargs):
        pass

class A1(a1_original.A1):
    def __init__(
      self,
      pybullet_client,
      sensors=None,
      xml_path = XML_PATH,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      self_collision_enabled=False,
      control_latency=0.002,
      on_rack=False,
      reset_at_current_position=False,
      reset_func_name="_mj_Reset",
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      motor_torque_limits=MAX_TORQUE,
      reset_time=1,
      allow_knee_contact=False,
      log_time_per_step=False,
      observation_noise_stdev=(0.0,) * 6,
      velocity_source=VelocitySource.PYBULLET,

  ):
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.num_motor = self.model.nu
        self.motor_dict = {}
        self.motor_names = []
        self.joint_angles = np.zeros(self.model.njnt)  # Углы шарниров
        self.joint_velocities = np.zeros(self.model.njnt)  # Скорости шарниров
        self.locker = Lock()
        self.executor = futures.ThreadPoolExecutor(max_workers=2)  # Два потока для viewer и simulation
        self.task_queue = Queue()  # Очередь задач для симуляции
        self.MIN_KD = 1
        self.MAX_KD = 10
        self._velocity = [0,0,0]
        self.motor_names = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint',
                       'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']

        self.motor_dict = {'FR_hip_joint': 0, 'FR_thigh_joint': 1, 'FR_calf_joint': 2, 'FL_hip_joint': 3, 'FL_thigh_joint': 4, 'FL_calf_joint': 5, 'RR_hip_joint': 6, 'RR_thigh_joint': 7,
              'RR_calf_joint': 8, 'RL_hip_joint': 9, 'RL_thigh_joint': 10, 'RL_calf_joint': 11}

        self._base_orientation = [1,0,0,0]
        self._reset_func = getattr(self, reset_func_name)
        super(A1, self).__init__(pybullet_client=pybullet_client, 
                                    reset_time=-1,
                                    velocity_source=False,
                                    reset_func_name=reset_func_name,
                                    motor_control_mode=a1_original.robot_config.MotorControlMode.POSITION
                                    )
        
        self.motor_kp = [
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
            HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
        ]
        self.motor_kd = [
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
            HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
        ]
        # self._joint_angle_lower_limits = np.array(
        #     [field.lower_bound for field in self.ACTION_CONFIG])
        # print(f"Test: {self._joint_angle_lower_limits}")
        # self._joint_angle_upper_limits = np.array(
        #     [field.upper_bound for field in self.ACTION_CONFIG])

        # Initialize sensors
        self._sensors = []
        if sensors is not None:
            self.SetAllSensors(sensors)
#    def __del__(self):
#      self.LogTimesteps()

    def _GetDefaultInitPosition(self):
        """Returns the init position of the robot.
    
        It can be either 1) origin (INIT_POSITION), 2) origin with a rack
        (INIT_RACK_POSITION), or 3) the previous position.
        """
        # If we want continuous resetting and is not the first episode.
        if self._reset_at_current_position and self._observation_history:
          x, y, _ = self.GetBasePosition()
          _, _, z = self.INIT_POSITION
          return [x, y, z]
    
        if self._on_rack:
          return self.INIT_RACK_POSITION
        else:
          return self.INIT_POSITION
    
    def _GetDefaultInitOrientation(self):
        """Returns the init position of the robot.
    
        It can be either 1) INIT_ORIENTATION or 2) the previous rotation in yaw.
        """
        # If we want continuous resetting and is not the first episode.
        if self._reset_at_current_position and self._observation_history:
          _, _, yaw = self.GetBaseRollPitchYaw()
          return  Rotation.from_euler('xyz', [0.0, 0.0, yaw], degrees=False).as_quat()
        return self.INIT_ORIENTATION
    
    def resetBasePositionAndOrientation(self, pos : NDArray, ori : NDArray, body_name: str = "trunk"):
        """Resets position and orientation of the robot.

        Args:
            pos - [x,y,z]
            ori - [x,y,z,w]
        """

        if not isinstance(body_name, str):
            raise ValueError("body_name must be a string")
        #получаем id главного тела собаки, в случае a1 - "trunk"
        body_id = self.model.body(body_name).id
        # Обновляем положение и ориентацию тела
        self.data.qpos[body_id * 7 : body_id * 7 + 3] = pos  # Положение (x, y, z)
        self.data.qpos[body_id * 7 + 3 : body_id * 7 + 7] = ori  # Ориентация (кватернион)
        mujoco.mj_step(self.model, self.data)

    def resetBaseVelocity(self, body_name : str = "trunk"):
       body_id = self.model.body(body_name).id
       self.data.qvel[body_id * 6 : body_id * 6 + 6] = 0
    
    def _mj_Reset(self, default_motor_angles=None, reset_time=3.0):
        """Reset robot to initial pose."""
        # First set the trunk position and orientation
        init_position = [0, 0, 0.3]  # x, y, height above ground
        init_orientation = [1, 0, 0, 0]  # quaternion [w, x, y, z] for identity rotation
        
        # Get trunk body ID
        trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        
        # Reset all positions and velocities to zero first
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # Set trunk position and orientation (first 7 values in qpos)
        self.data.qpos[0:3] = init_position
        self.data.qpos[3:7] = init_orientation
        
        # Now set the motor angles
        if default_motor_angles is None:
            default_motor_angles = INIT_MOTOR_ANGLES
        
        # Set joint positions - starting at index 7 since trunk uses 0-6
        for i, motor_name in enumerate(self.motor_names):
            joint_id = self.model.joint(motor_name).id
            qpos_index = 7 + i  # Offset by 7 to skip trunk position/orientation
            #print(f"Setting {motor_name} (id: {joint_id}) at qpos[{qpos_index}] to {default_motor_angles[i]}")
            self.data.qpos[qpos_index] = default_motor_angles[i]
        
        # Forward the simulation to apply changes
        mujoco.mj_forward(self.model, self.data)
        
        # Debug prints
        #print("\nAfter reset:")
        #print(f"Trunk position: {self.data.qpos[0:3]}")
        #print(f"Trunk orientation: {self.data.qpos[3:7]}")
        for i, motor_name in enumerate(self.motor_names):
            qpos_index = 7 + i

    def get_joint_states_mujoco(self, model, data, joint_dict: dict | None = None):
        """
        Аналог getJointStates из PyBullet для MuJoCo, включая реактивные силы.

        Параметры:
        - model: объект MjModel (описание модели)
        - data: объект MjData (текущее состояние симуляции)
        - joint_names: список имён суставов (если None, берутся все)

        Возвращает:
        - Словарь {joint_name: (position, velocity, torques)}
        """

        if not isinstance(joint_dict, dict):
            joint_dict = self.motor_dict
            #joint_names = model.motors.names.decode('utf-8').split('\x00') 
            #print("Not so good: ")

        joint_states =[]
        for joint_index in self.motor_dict.values():

            qpos_index = model.jnt_qposadr[joint_index]
            qvel_index = model.jnt_dofadr[joint_index]

            position = data.qpos[qpos_index] if qpos_index != -1 else None
            velocity = data.qvel[qvel_index] if qvel_index != -1 else None
            reaction_force = tuple(data.qfrc_constraint[qvel_index:qvel_index+6] if qvel_index != -1 else None)
            actuator_torque = data.qfrc_actuator[qvel_index] if qvel_index != -1 else None

            joint_states.append((position, velocity, reaction_force, actuator_torque))

        return tuple(joint_states)
    def convert_to_torque(self, motor_commands, motor_angles, motor_velocities, true_motor_velocities, motor_control_mode=None):
        """Convert the commands (position control or pwm control) to torque."""
        pwm = -1 * self.motor_kp * (motor_angles - motor_commands) - self.motor_kd * motor_velocities
        pwm = np.clip(pwm, -1.0, 1.0)
        return self._motor_model._convert_to_torque_from_pwm(pwm)
    
    def ApplyAction(self, motor_commands, motor_control_mode=None, motor_kp=1.0, motor_kd=0.02,):
        """Apply the motor commands using the motor model.

        Args:
            motor_commands: np.array. Can be motor angles, torques, or hybrid commands.
            motor_control_mode: A MotorControlMode enum.
        """
        self.last_action_time = self._state_action_counter * self.time_step
        control_mode = motor_control_mode if motor_control_mode is not None else self._motor_control_mode

        motor_commands = np.asarray(motor_commands)
        q, qdot = self._GetPDObservation()
        qdot_true = self.GetTrueMotorVelocities()
        
        # Get torque commands from the motor model
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
                                    motor_commands, q, qdot, qdot_true, control_mode)
        
        self._observed_motor_torques = observed_torque
        
        # Transform into the motor space when applying the torque
        self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)
        
        # Clip the commands
        minimum, maximum = -np.pi, np.pi

        # Apply commands in the correct order
        self.data.ctrl[:] = motor_commands

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetTrueMotorAngles())
        observation.extend(self.GetTrueMotorVelocities())
        observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetTrueBaseOrientation()) #нормально работает, обновляется из моей перезаписанной функции
        observation.extend(self.GetTrueBaseRollPitchYawRate())
        return observation

    def GetTrueMotorAngles(self):
        return np.array([self.data.qpos[self.motor_dict[name]] for name in self.motor_names])
        # print(f"{motor_qpos} : GettrueMOTORPOS")

    def GetTrueMotorVelocities(self):
        return np.array([self.data.qvel[self.motor_dict[name]] for name in self.motor_names])
        # print(motor_qvel)
        # motor_qvel = motor_qvel[-1]
        # print(f"{motor_qvel} : GettrueMOTORVEL")

    def GetTrueMotorTorques(self):
        return np.array([self.data.qfrc_applied[self.motor_dict[name]] for name in self.motor_names])#xfrc
    
    def getBasePositionAndOrientation(self):
        """
        Returns tuple of bas position, orientation and relarive orientation
        """
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'A1')
        init_orientation_inv = [0, 0, 0, 1]
        base_position = self.data.xpos[robot_id]
        base_orientation = self.data.xquat[robot_id]
        relative_orientation = self.quat_mul(base_orientation, init_orientation_inv)
        #print("Base Position:", base_position)
        #print("Base Orientation:", base_orientation)
        #print("Relative Orientation:", relative_orientation)
        return base_position, base_orientation, relative_orientation

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        #print("FROM ReceiveObservation")
        self._joint_states = self.get_joint_states_mujoco(self.model, self.data, joint_dict=self.motor_dict)
        self._base_position, _, self._base_orientation = self.getBasePositionAndOrientation() 
        self._prev_velocity = self._velocity
        self._velocity = np.array(self.data.qvel[:3])
        self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self._GetControlObservation()
        self._last_state_time = self._state_action_counter * self.time_step

    def GetTrueBaseRollPitchYawRate(self):
        angular_velocity = self.data.qvel[3:6]
        orientation = self.GetTrueBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angular_velocity,
                                                     orientation)

    def quat_mul(self, q1, q2):
    # """ Умножение двух кватернионов q1 и q2. """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ]


    def start(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Запуск viewer и simulation в потоках
        #self.executor.submit(self._viewer_loop)

    # def _viewer_loop(self):
    #     while self.viewer.is_running():
    #         with self.locker:
    #             self.viewer.sync()
    #         time.sleep(0.1)
            
    def add_task(self, task, *args):
        """Добавляет задачу в очередь."""
        self.task_queue.put((task, args))

    def _StepInternal(self, action, motor_control_mode):
        #print("STEP_INTERNAL CALLED")
        self.ApplyAction(action, motor_control_mode)
        mujoco.mj_step(self.model, self.data)  # Шаг симуляции
        self.viewer.sync()
        self.ReceiveObservation()
        self._state_action_counter += 1

    def Step(self, action, control_mode=None):
      """Steps simulation."""
      #print("STEP CALLED")
      if self._enable_action_filter:
        action = self._FilterAction(action)
      if control_mode==None:
        control_mode = self._motor_control_mode
      for i in range(self._action_repeat):
        proc_action = self.ProcessAction(action, i)
        self._StepInternal(proc_action, control_mode)
        self._step_counter += 1 
      self._last_action = action
      
    def Brake(self):
      # Braking on the real robot has more resistance than this.
      # Call super to avoid doing safety checks while braking.
      self._StepInternal(
          np.zeros((self.num_motors,)),
          motor_control_mode=robot_config.MotorControlMode.TORQUE)
      self.LogTimesteps()

    def _SettleDownForReset(self, default_motor_angles=None, reset_time=10): #default_motor_angles=np.array(_DEFAULT_HIP_POSITIONS).flatten
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
          #print(_)
          self._StepInternal(
              INIT_MOTOR_ANGLES,
              motor_control_mode=robot_config.MotorControlMode.POSITION)

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
              self._StepInternal(
                default_motor_angles,
                motor_control_mode=robot_config.MotorControlMode.POSITION)


    def ResetPose(self, add_constraint=1):
        """
        Сбрасываем позу робота в MuJoCo в предопределенное начальное положение.
        """
        del add_constraint
        # print("Available joint names in the model:", [self.model.joint(i).name for i in range(self.model.njnt)])
        # print("self.motor_names:", self.motor_names) #  инициализация motor_names
        # Сброс углов суставов
        for motor_name in self.motor_names:
            joint_index = self.model.joint(motor_name).id # Получение id сустава
            if joint_index is None:  # Проверка на существование сустава иначе проблемы могут быть 
                print(f"Warning: Joint '{motor_name}' not found in the model.")
                continue

            i = self.motor_names.index(motor_name)

            if "hip" in motor_name:
                angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
            elif "thigh" in motor_name:
                angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
            elif "calf" in motor_name:
                angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
            else:
                raise ValueError(f"The name {motor_name} is not recognized as a motor joint.")
            self.data.qpos[joint_index] = angle

        # Сброс скоростей data.qvel
        for motor_name in self.motor_names:
            joint_index = self.model.joint(motor_name).id
            if joint_index is None:
                continue
            self.data.qvel[joint_index] = 0.0 # Установка нулевой скорости

        # #  можно сбросить положение и ориентацию всего робота:
        # # установка позиции:
        # self.data.qpos[:3] = [0.0, 0.0, 0.3]  # x, y, z - задайте начальное положение.
        # # ориентация (кватернион):
        # self.data.qpos[3:7] = [1, 0, 0, 0] # w, x, y, z (единичный кватернион - начальная ориентация)
        # # Сброс скоростей тела робота :
        # self.data.qvel[:3] = [0, 0, 0] # линейная скорость
        # self.data.qvel[3:6] = [0, 0, 0] # угловая скорость

        #  Обновляем симуляцию (важно для применения изменений):
        mujoco.mj_forward(self.model, self.data)

    def _CollapseReset(self, default_motor_angles, reset_time):
        """Sets joint torques to 0, then moves joints within bounds."""
        del default_motor_angles
        del reset_time
        print("FROM _CollapseReset")
        # Important to fill the observation buffer.
        self.ReceiveObservation()
        # Spend 1 second collapsing.
        half_steps_to_reset = int(0.5 / self.time_step)
        for _ in range(half_steps_to_reset):
          self.Brake()
        for _ in range(half_steps_to_reset):
          self._StepInternal(
              np.zeros((self.num_motors,)),
              motor_control_mode=robot_config.MotorControlMode.TORQUE)
        self._SafeJointsReset()

    def _AreAnglesInvalid(self):
      return (any(self.GetTrueMotorAngles() > self._joint_angle_upper_limits - 0.03) or
              any(self.GetTrueMotorAngles() < self._joint_angle_lower_limits + 0.03))
    
    def _SafeJointsReset(self, default_motor_angles=None, reset_time=1):
        """Moves joints within bounds safely using keyframe data."""
        print("Stand up reset called!", reset_time, default_motor_angles)

        if reset_time <= 0:
            return

        # Extract default angles from keyframe
        if not default_motor_angles:
            try:
                # Get qpos from 'home' keyframe (first 7 values are base pos/orientation)
                keyframe = self.model.keyframe('home')
                full_qpos = keyframe.qpos
                # Extract only joint angles (skip first 7 values)
                default_motor_angles = full_qpos[7:]  
            except Exception as e:
                print(f"Error loading keyframe: {e}")
                default_motor_angles = self.GetDefaultInitJointPose()

        current_motor_angles = self.GetMotorAngles()
        standup_time = 1.5

        for t in np.arange(0, standup_time, self.time_step * self._action_repeat):
            blend_ratio = min(t / standup_time, 1)
            action = blend_ratio * default_motor_angles + (1 - blend_ratio) * current_motor_angles
            self.data.ctrl[:] = action
        
            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def _ClipMotorAngles(self, desired_angles, current_angles):
        if self._enable_clip_motor_commands:
            angle_ub = np.minimum(self._joint_angle_upper_limits + MOTOR_OFFSETS,
                            current_angles + MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
            angle_lb = np.maximum(self._joint_angle_lower_limits + MOTOR_OFFSETS,
                            current_angles - MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
        else:
            angle_ub = self._joint_angle_upper_limits
            angle_lb = self._joint_angle_lower_limits
        return np.clip(desired_angles, angle_lb, angle_ub)
#MOTOR_MAXS, MOTOR_MINS
    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=0.0):
        """Reset the minitaur to its initial states.

        Args:
          reload_urdf: Whether to reload the urdf file. If not, Reset() just place
            the minitaur back to its starting position.
          default_motor_angles: The default motor angles. If it is None, minitaur
            will hold a default pose (motor angle math.pi / 2) for 100 steps. In
            torque control mode, the phase of holding the default pose is skipped.
          reset_time: The duration (in seconds) to hold the default motor angles. If
            reset_time <= 0 or in torque control mode, the phase of holding the
            default pose is skipped.
        """
        print("RESET CALLED")
        self._reset_func(default_motor_angles, reset_time)

        self._overheat_counter = np.zeros(self.num_motors)
        self._motor_enabled_list = [True] * self.num_motors
        self._observation_history.clear()
        self._step_counter = 0
        self._state_action_counter = 0
        self._is_safe = True
        self._last_action = None
        self.ReceiveObservation()


        self._position_at_reset, _, self._quat_at_reset = self.getBasePositionAndOrientation()
        self._velocity = np.zeros((3,))
        self._prev_velocity = np.zeros((3,))
        self._accelerometer_reading = np.zeros((3,))
        self._last_state_time = 0

    def GetBaseVelocity(self):
        """Get the linear velocity of the robot's base.
        Returns:
            The current velocity of the robot's base as [vx, vy, vz].
        """
        return self.data.qvel[:3].copy()

    def GetFootContacts(self):
        """Returns a list of booleans indicating whether each foot is in contact.
        
        Returns:
            List of 4 booleans, one for each foot (FR, FL, RR, RL).
        """
        foot_links = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        contacts = []
        
        for foot in foot_links:
            # Get the body ID for the foot
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot)
            # Check if this body has any contacts
            has_contact = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == body_id or contact.geom2 == body_id:
                    has_contact = True
                    break
            contacts.append(has_contact)
            
        return contacts

    def GetDefaultInitPosition(self):
        """Returns default initial position of the robot."""
        return self.INIT_POSITION

    def GetDefaultInitOrientation(self):
        """Returns default initial orientation of the robot as quaternion."""
        return self.INIT_ORIENTATION

    def GetDefaultInitJointPose(self):
        """Returns default initial joint angles."""
        return INIT_MOTOR_ANGLES

    def HoldCurrentPose(self):
        """Holds the current pose of the robot using position control."""
        current_angles = self.GetMotorAngles()
        self.Step(current_angles, robot_config.MotorControlMode.POSITION)

    def _ValidateMotorStates(self):
        """Checks if the motor states are valid.
        
        Returns:
            True if the states are valid, False otherwise.
        """
        motor_angles = self.GetTrueMotorAngles()
        motor_velocities = self.GetTrueMotorVelocities()
        motor_torques = self.GetTrueMotorTorques()

        # Check if angles are within limits
        if any(motor_angles > self._joint_angle_upper_limits) or \
           any(motor_angles < self._joint_angle_lower_limits):
            return False

        # Check if velocities are within limits
        if any(abs(motor_velocities) > MAX_JOINT_VELOCITY):
            return False

        # Check if torques are within limits
        if any(abs(motor_torques) > MAX_TORQUE):
            return False

        return True

    def LogTimesteps(self):
        """Logs timestep information if enabled."""
        #if not self._log_time_per_step:
        #    return
        ## Add your logging logic here if needed
        pass

    def SetAllSensors(self, sensors):
        """Set all sensors to this robot and move the ownership to this robot.
        
        Args:
            sensors: a list of sensors to this robot.
        """
        for s in sensors:
            s.set_robot(self)
        self._sensors = sensors

    def GetAllSensors(self):
        """Get all sensors attached to this robot.
        
        Returns:
            The list of all sensors.
        """
        print(self._sensors)
        return self._sensors

    def GetTrueBaseRollPitchYaw(self):
        """Get the robot's base orientation in euler angle in the world frame.

        Returns:
            A tuple (roll, pitch, yaw) of the base in world frame.
        """
        # Get quaternion from MuJoCo
        quat = self.data.qpos[3:7]  # [w, x, y, z] format
        
        # Use MuJoCo's built-in quaternion to euler conversion
        euler = np.zeros(3)
        mujoco.mju_quat2euler(euler, quat)
        
        return euler  # Returns [roll, pitch, yaw]

    def MapToMinusPiToPi(self, angles):
      """Maps a list of angles to [-pi, pi].

      Args:
        angles: A list of angles in rad.

      Returns:
        A list of angle mapped to [-pi, pi].
      """
      mapped_angles = copy.deepcopy(angles)
      for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], TWO_PI)
        if mapped_angles[i] >= math.pi:
          mapped_angles[i] -= TWO_PI
        elif mapped_angles[i] < -math.pi:
          mapped_angles[i] += TWO_PI
      return mapped_angles
    
    def GetMotorAngles(self):
      """Gets the eight motor angles.

      This function mimicks the noisy sensor reading and adds latency. The motor
      angles that are delayed, noise polluted, and mapped to [-pi, pi].

      Returns:
        Motor angles polluted by noise and latency, mapped to [-pi, pi].
      """
      motor_angles = self._AddSensorNoise(
          np.array(self._control_observation[0:self.num_motors]),
          self._observation_noise_stdev[0])
      #print("motor_angles: ", self.MapToMinusPiToPi(motor_angles))
      return self.MapToMinusPiToPi(motor_angles)

    def print_xml_defaults(self):
        """Print the default angles from XML for comparison."""
        tmp_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, tmp_data, 0)
        
        print("\nMuJoCo XML Default Angles:")
        for i, motor_name in enumerate(self.motor_names):
            joint_id = self.model.joint(motor_name).id
            angle = tmp_data.qpos[joint_id]
            print(f"{motor_name}: {angle:.3f}")

if __name__ == '__main__':
    pyb.connect(a1_original.pyb.DIRECT)
    robot = A1(pybullet_client=pyb)
    robot.ReceiveObservation()
    robot.Reset()
    time.sleep(1)

    # Test each joint individually
    for i in range(11):
        print(f"\nTesting joint {i}")
        print(f"Motor name: {robot.motor_names[i]}")
        
        # Create a test action vector (all zeros except one joint)
        test_action = -np.ones(12)*0.0
        for _ in range(10):
            test_action[i] = np.random.randint(-3, 3)
            print(test_action[i], "test_action",i, "joint name", robot.motor_names[i])
            robot.Step(test_action)
        robot._SafeJointsReset()
        time.sleep(20)
            #time.sleep(0.01)  # Set only one joint to move
        
        # Get initial position
        init_angles = robot.GetTrueMotorAngles()
        #print(f"Initial angles: {init_angles}")
        
        # Apply action
        for i in range(10):
            robot.Step(test_action)        
        # Get new position
        #new_angles = robot.GetTrueMotorAngles()
        #print(f"New angles: {new_angles}")
        #print(f"Difference: {new_angles - init_angles}")
        
        time.sleep(1)  # Wait to observe movement
        
        # Reset to initial position
        robot.Reset()
        time.sleep(1)