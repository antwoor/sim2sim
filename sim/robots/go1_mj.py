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
from motion_imitation.robots import go1 as go1#a1
XML_PATH = os.path.join(parentdir , "sim/robots/unitree_go1/scene.xml")
_UNIT_QUATERNION = (1, 0, 0, 0)
ABDUCTION_D_GAIN = go1.ABDUCTION_D_GAIN
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

class Go1_mj(go1.Go1):
    def __init__(
      self,
      xml_path = XML_PATH,

  ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_motor = self.model.nu
        self.motor_dict = {}
        self.motor_names = []
        self.kd = 1
        self.kp = 1 
        self.joint_angles = np.zeros(self.model.njnt)  # Углы шарниров
        self.joint_velocities = np.zeros(self.model.njnt)  # Скорости шарниров
        self.locker = Lock()
        self.executor = futures.ThreadPoolExecutor(max_workers=2)  # Два потока для viewer и simulation
        self.task_queue = Queue()  # Очередь задач для симуляции
        self.MIN_KD = 1
        self.MAX_KD = 10
        self._velocity = [0,0,0]
        self.motor_dict = {'FR_hip': 0, 'FR_thigh': 1, 'FR_calf': 2, 'FL_hip': 3, 'FL_thigh': 4, 'FL_calf': 5, 'RR_hip': 6, 'RR_thigh': 7, 
              'RR_calf': 8, 'RL_hip': 9, 'RL_thigh': 10, 'RL_calf': 11}

        self.motor_names = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh',
                       'FL_calf', 'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']

        self._base_orientation = [1,0,0,0]
        super(Go1_mj, self).__init__(pybullet_client=go1.pyb, 
                                    reset_time=-1,
                                    velocity_source=False,
                                    reset_func_name='_mj_Reset',
                                    motor_control_mode=go1.robot_config.MotorControlMode.TORQUE)

    def _mj_Reset():
        pass
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

        if isinstance(joint_dict, dict):
            print("Good: ")
        else:
            joint_dict = self.motor_dict
            #joint_names = model.motors.names.decode('utf-8').split('\x00') 
            print("Not so good: ")
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
    
    def apply_action(self, desired_qpos):

        for i in range(self.model.njnt):
            qpos_id = self.model.jnt_qposadr[i]
            qvel_id = self.model.jnt_dofadr[i]

        # Массивы для углов и скоростей всех шарниров
        # Заполняем массивы
            self.joint_angles = self.data.qpos[qpos_id] # Угол текущего шарнира
            self.joint_velocities = self.data.qvel[qvel_id]  # Скорость текущего шарнира
            #print(f"{self.joint_angles}: Angles (qpose) = {self.joint_angles}")
            #print(f"{self.joint_velocities}: Velocities (qpose) = {self.joint_velocities}")

        if self.kd is None:


            kd = (desired_qpos[-1] + 1) / 2 * (self.MAX_KD - self.MIN_KD) + self.MIN_KD
            desired_qpos = desired_qpos[:-1]
        else:
            kd = self.kd

        action = self.kp * (desired_qpos - self.data.qpos[qpos_id]) - kd * self.data.qvel[qvel_id]
        minimum, maximum = -np.pi, np.pi
        action = np.clip(action, minimum, maximum)

        for i in range(self.num_motor):
            self.data.ctrl[i] = action[i]

    def GetTrueObservation(self):
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~check motor id~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # # print(self.num_motor) - 12

        # for i in range(self.model.nu):
        #     motor_name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i)
        #     self.motor_names.append(motor_name)
        #     joint_id = []
        #     # joint_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,motor_names)
        #     # print(joint_id)
        #     self.motor_qpos[motor_name] = i
        # ДЛЯ НОВОГО РОБОТА
        # print(self.motor_qpos) - НУЖНО заново раскомитить и вставить в соотв словарь
        # print(self.motor_names)- НУЖНО заново раскомитить и вставить в соотв список

        observation = []
        observation.extend(self.GetTrueMotorAngles())
        observation.extend(self.GetTrueMotorVelocities())
        observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetTrueBaseOrientation()) #нормально работает, обновляется из моей перезаписанной функции
        observation.extend(self.GetTrueBaseRollPitchYawRate())
        return observation
        # print(motor_qpos_id)_base_orientation
        # motor_name = "FL_hip"
        # motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        # print(motor_id)  


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
        print("Base Position:", base_position)
        print("Base Orientation:", base_orientation)
        print("Relative Orientation:", relative_orientation)
        return base_position, base_orientation, relative_orientation

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
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
        self.executor.submit(self._viewer_loop)
        self.executor.submit(self._simulation_loop)

    def _simulation_loop(self):
        while self.viewer.is_running():
            #step_start = time.perf_counter()
            with self.locker:
                mujoco.mj_step(self.model, self.data)  # Шаг симуляции
                # Обработка задач из очереди
                try:
                    while not self.task_queue.empty():
                        task, args = self.task_queue.get()
                        task(*args)
                except Empty:
                    pass

    def _viewer_loop(self):
        while self.viewer.is_running():
            with self.locker:
                self.viewer.sync()
            time.sleep(0.01)
            
    def add_task(self, task, *args):
        """Добавляет задачу в очередь."""
        self.task_queue.put((task, args))

if __name__ == '__main__':
    go1.pyb.connect(go1.pyb.DIRECT)
    robot = Go1_mj()
    #robot.GetTrueObservation()
    robot.start()
    #print(robot.get_joint_states_mujoco(robot.model, robot.data))
    #time.sleep(2)
    print("=============================")
    #print(robot.GetTrueMotorAngles())
    robot.GetTrueMotorAngles()
    robot.ReceiveObservation()
    robot.Step(action=np.random.randint(-3, 3, 12))
    robot._StepInternal(action=np.random.randint(-3, 3, 12))
    robot.GetTrueBaseOrientation()
    robot.GetFootContacts()
    #robot._SafeJointsReset()
    robot.ResetPose(12)
    robot._GetMotorNames()
    robot.GetDefaultInitPosition()
    robot.GetDefaultInitOrientation()
    robot.GetDefaultInitJointPose()
    robot.Reset()
    #robot._CollapseReset()
    #robot._ClipMotorAngles() #TODO впадлу
    #robot._ClipMotorCommands() #TODO впадлу
    robot.Brake()
    robot.HoldCurrentPose()
    robot._ValidateMotorStates()
    robot.GetBaseVelocity()
    robot.LogTimesteps()
    print("+++++++++++++++++++++")
    print(robot._base_orientation)
    print(robot._base_position)
    print(robot._observation_history[0])
    print(len(robot._observation_history[0]))
    #robot.ComputeMotorAnglesFromFootLocalPosition() #TODO впадлу
    #robot.GetFootPositionsInBaseFrame() #TODO впадлу
    #robot.ComputeJacobian() #TODO впадлу



    #time.sleep(10)
    #print(robot.get_joint_states_mujoco(robot.model, robot.data, joint_dict=robot.motor_dict))
    #print("======================================")
    #print(robot.get_joint_states_mujoco(robot.model, robot.data))
    #print("++++++++++++++++++++++++++++++++++++++")
    #print("size: ", len(robot.get_joint_states_mujoco(robot.model, robot.data)))    
    #print("size: element ", len(robot.get_joint_states_mujoco(robot.model, robot.data)[0]))  
    #joint_states = robot.get_joint_states_mujoco(robot.model, robot.data)
    #print("first:  ", joint_states[0])    
    #for _ in range(1000):
    #    random_action = np.random.randint(-3, 3, 12)
    #    robot.add_task(robot.apply_action, np.random.randint(-3, 3, 12))
    #    robot.add_task(robot.GetTrueObservation)
    #    time.sleep(0.01)