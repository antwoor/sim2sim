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

"""Pybullet simulation of a Go1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import collections.abc
import enum
import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error
import time

import asyncio

from motion_imitation.robots import a1_robot_velocity_estimator
from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur  # Is it important yet?
# from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config
#TODO: Рубен, пожалуйсат, сделай так, чтобы импортировалось и при запуске файла появлялось окно маджоки, заранее большое спасибо!) import mujoco
#TODO: Рубен, пожалуйсат, сделай так, чтобы импортировалось и при запуске файла появлялось окно маджоки, заранее большое спасибо!) import sim.dmcgym
#TODO: Рубен, пожалуйсат, сделай так, чтобы импортировалось и при запуске файла появлялось окно маджоки, заранее большое спасибо!) from sim import robots

import mujoco
from sim import robots


NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2  # TODO
MAX_TORQUE = np.array([28.7, 28.7, 40] * NUM_LEGS)
MAX_JOINT_VELOCITY = np.array([30.1,30.1,20.06]*NUM_LEGS)

DEFAULT_HIP_POSITIONS = (
     (0.1, 0.8, -1.5), #(0.179, 0.497, 0.360)
     (-0.1, 0.8, -1.5),#(-0.594, 0.219, 0.602)
     ( 0.1, 1.0, -1.5),#( 0.543, 0.539, 0.436)
     (-0.1, 1.0, -1.5),#(-0.241, 0.440, 0.210)
 )

COM_OFFSET = -np.array([0.0223, 0.002, -0.0005]) # Test Param
HIP_OFFSETS = np.array([[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.],
                        [-0.1881, -0.04675, 0.], [-0.1881, 0.04675, 0.]
                        ]) + COM_OFFSET 

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS) # maybe add from DEFAULT_HIP_POSITION

MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
MOTOR_MINS = np.array([
    -1.047,
    -0.663,
    -2.721,
] * 4)

MOTOR_MAXS = np.array([
    1.047,
    2.966,
    -0.837,
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
UPPER_NAME_PATTERN = re.compile(r"\w+_thigh_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_calf_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = os.path.join(parentdir, "motion_imitation/utilities/go1/urdf/go1.urdf")  ## TODO: add Go1.urdf motion_imitation/utilities/a1/a1.urdf

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

# Empirical values from real A1.#TODO
ACCELEROMETER_VARIANCE = 0.03059
JOINT_VELOCITY_VARIANCE = 0.006206


class VelocitySource(enum.Enum):
  PYBULLET = 0
  IMU_FOOT_CONTACT = 1
  MJ = 3

class Go1(minitaur.Minitaur):
  """A simulation for the Go1 robot."""

  # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
  # doesn't seem to matter much. However, these values should be better tuned
  # when the replan frequency is low (e.g. using a less beefy CPU).
  MPC_BODY_MASS = 5.204 * 2 #108 / 9.8
  MPC_BODY_INERTIA = np.array((0.0168128557, 0, 0, 
                                      0, 0.063009565, 0, 
                                      0, 0, 0.0716547275)) * 5.
  MPC_BODY_HEIGHT = 0.30
  MPC_VELOCITY_MULTIPLIER = 0.5
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name=key, upper_bound=hi, lower_bound=lo)
      for key, hi, lo in zip(MOTOR_NAMES, MOTOR_MAXS, MOTOR_MINS)]
  INIT_RACK_POSITION = [0, 0, 1]
  INIT_POSITION = [0, 0, 0.30]
  INIT_ORIENTATION = (0, 0, 0, 1)
  # Joint angles are allowed to be JOINT_EPSILON outside their nominal range.
  # This accounts for imprecision seen in either pybullet's enforcement of joint
  # limits or its reporting of joint angles.
  JOINT_EPSILON = 0.1

  def __init__(
      self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      self_collision_enabled=False,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      reset_at_current_position=False,
      reset_func_name="_PybulletReset",
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      motor_torque_limits=MAX_TORQUE,
      reset_time=1,
      allow_knee_contact=False,
      log_time_per_step=False,
      observation_noise_stdev=(0.0,) * 6,
      velocity_source=VelocitySource.MJ,
      sim2sim="mujoco"
  ):
    """Constructor.

    Args:
      observation_noise_stdev: The standard deviation of a Gaussian noise model
        for the sensor. It should be an array for separate sensors in the
        following order [motor_angle, motor_velocity, motor_torque,
        base_roll_pitch_yaw, base_angular_velocity, base_linear_acceleration]
      velocity_source: How to determine the velocity returned by
        self.GetBaseVelocity().
    """
    self.running_reset_policy = False
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]
    self._joint_angle_lower_limits = np.array(
        [field.lower_bound for field in self.ACTION_CONFIG])
    self._joint_angle_upper_limits = np.array(
        [field.upper_bound for field in self.ACTION_CONFIG])
    if log_time_per_step:
      self._timesteps = []
    else:
      self._timesteps = None
    self._last_step_time_wall = 0
    self._currently_resetting = False
    self._max_vel = 0
    self._max_tau = 0
    self._velocity_estimator = None

    if velocity_source is VelocitySource.IMU_FOOT_CONTACT:
     self._velocity_estimator = a1_robot_velocity_estimator.VelocityEstimator(
         robot=self,
         accelerometer_variance=ACCELEROMETER_VARIANCE,
         sensor_variance=JOINT_VELOCITY_VARIANCE)
    if sim2sim == "mujoco":
      self.robot = robots.A1()
      self.robot.start()
    super().__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        self_collision_enabled=self_collision_enabled,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        motor_torque_limits=motor_torque_limits,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        observation_noise_stdev=observation_noise_stdev,
        on_rack=on_rack,
        reset_at_current_position=reset_at_current_position,
        reset_func_name=reset_func_name,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time)


  def __del__(self):
    self.LogTimesteps()

  def _LoadRobotURDF(self):
    a1_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "hip_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "thigh_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "calf_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self): #TODO: в этом методе чекаются все джойнты и убираются проблемы с лишними и none джойнтами и этот метод нужно переписать на  xml mujoco и желательно не убирать реализацию для urdf чтобы ничего не похерилось
    """Build the link Ids from its name in the URDF file.
    
    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._hip_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._lower_link_ids = []
    self._foot_link_ids = []
    self._imu_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        self._hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        self._lower_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        #assert self._urdf_filename == URDF_WITH_TOES
        self._foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        self._imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._lower_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)

    #assert len(self._foot_link_ids) == NUM_LEGS
    self._hip_link_ids.sort()
    self._motor_link_ids.sort()
    self._lower_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()
    print("hip links", self._hip_link_ids)
    print("motors", self._motor_link_ids)
    print("lower_link_ids", self._lower_link_ids)
    print("_foot_link_ids", self._foot_link_ids)
    print("_leg_link_ids", self._leg_link_ids)

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    if motor_control_mode is None:
      motor_control_mode = self._motor_control_mode
    motor_commands = self._ClipMotorCommands(motor_commands, motor_control_mode)
    super().ApplyAction(motor_commands, motor_control_mode)
    self.robot.add_task(self.robot.apply_action, motor_commands)

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

  def _ClipMotorCommands(self, motor_commands, motor_control_mode):
    """Clips commands to respect any set joint angle and torque limits.

    Always clips position to be within ACTION_CONFIG. If
    self._enable_clip_motor_commands, also clips positions to be within
    MAX_MOTOR_ANGLE_CHANGE_PER_STEP of current positions.
    Always clips torques to be within self._motor_torque_limits (but the torque
    limits can be infinity).

    Args:
      motor_commands: np.array. Can be motor angles, torques, or hybrid.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      Clipped motor commands.
    """
    if motor_control_mode == robot_config.MotorControlMode.TORQUE:
      return np.clip(motor_commands, -1 * self._motor_torque_limits, self._motor_torque_limits)
    if motor_control_mode == robot_config.MotorControlMode.POSITION:
      return self._ClipMotorAngles(
          desired_angles=motor_commands,
          current_angles=self.GetTrueMotorAngles())
    if motor_control_mode == robot_config.MotorControlMode.HYBRID:
      # Clip angles
      angles = motor_commands[np.array(range(NUM_MOTORS)) * 5]
      clipped_positions = self._ClipMotorAngles(
          desired_angles=angles,
          current_angles=self.GetTrueMotorAngles())
      motor_commands[np.array(range(NUM_MOTORS)) * 5] = clipped_positions
      # Clip torques
      torques = motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4]
      clipped_torques = np.clip(torques, -1 * self._motor_torque_limits, self._motor_torque_limits)
      motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4] = clipped_torques
      return motor_commands

  def _ValidateMotorStates(self):
    # Check torque.
    if any(np.abs(self.GetTrueMotorTorques()) > MAX_TORQUE ):
      raise robot_config.SafetyError(
          "Torque limits exceeded\ntorques: {}".format(
              self.GetTrueMotorTorques()))

    # Check joint velocities.
    if any(np.abs(self.GetTrueMotorVelocities()) > MAX_JOINT_VELOCITY):
      raise robot_config.SafetyError(
          "Velocity limits exceeded\nvelocities: {}".format(
              self.GetTrueMotorVelocities()))

    # Joints often start out of bounds (in sim they're 0 and on real they're
    # slightly out of bounds), so we don't check angles during reset.
    if self._currently_resetting or self.running_reset_policy:
      return
    # Check joint positions.
    # if (any(self.GetTrueMotorAngles() > (self._joint_angle_upper_limits +
    #                                     self.JOINT_EPSILON)) or
    #    any(self.GetTrueMotorAngles() < (self._joint_angle_lower_limits -
    #                                     self.JOINT_EPSILON))):
    #  raise robot_config.SafetyError(
    #      "Joint angle limits exceeded\nangles: {}".format(
    #          self.GetTrueMotorAngles()))

  def _StepInternal(self, action, motor_control_mode=None):
    if self._timesteps is not None:
      now = time.time()
      self._timesteps.append(now - self._last_step_time_wall)
      self._last_step_time_wall = now
    if not self._is_safe:
      return
    super()._StepInternal(action, motor_control_mode)
    # real world
    try:
      self._ValidateMotorStates()
    except robot_config.SafetyError as e:
      print(e)
      self.Brake()
      self._is_safe = False

  def ReceiveObservation(self):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    self._joint_states = self._pybullet_client.getJointStates(
        self.quadruped, self._motor_id_list)
    self._base_position, orientation = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    # Computes the relative orientation relative to the robot's
    # initial_orientation.
    _, self._base_orientation = self._pybullet_client.multiplyTransforms(
        positionA=[0, 0, 0],
        orientationA=orientation,
        positionB=[0, 0, 0],
        orientationB=self._init_orientation_inv)
    self._prev_velocity = self._velocity
    self._velocity = np.array(
        self._pybullet_client.getBaseVelocity(self.quadruped)[0])
    self._UpdateBaseAcceleration()
    self._observation_history.appendleft(self.GetTrueObservation())
    self._control_observation = self._GetControlObservation()
    self._last_state_time = self._state_action_counter * self.time_step
    if self._velocity_estimator:
      self._velocity_estimator.update(self.GetTimeSinceReset())

  def GetBaseVelocity(self): #TODO изменить на скорость туловища из муджоко должен быть вектор из трёх эелментов, как в пб, скорее всего [x,y,z]
    if self._velocity_estimator:
      return self._velocity_estimator.estimated_velocity
    return super().GetBaseVelocity()

  def LogTimesteps(self):
    if self._timesteps is None or not len(self._timesteps):
      return
    timesteps = np.asarray(self._timesteps[1:])
    print('=====\nTimestep stats (secs)\nlen: ', len(timesteps), '\nmean: ',
          np.mean(timesteps), "\nmin: ", np.min(timesteps), "\nmax: ",
          np.max(timesteps), "\nstd: ", np.std(timesteps), "\n=====")

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants

if __name__ == '__main__':
  pyb.connect(pyb.DIRECT)
  quadrupedal = Go1(pybullet_client =pyb)
  while True:
    quadrupedal.ApplyAction(motor_commands=np.random.randint(-3,3,12), motor_control_mode=robot_config.MotorControlMode.POSITION)
  # print(robot.GetTrueMotorAngles())
  #print(robot._BuildUrdfIds())
  

    