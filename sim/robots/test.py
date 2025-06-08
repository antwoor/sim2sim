# from dm_control import suite
# from dm_control.viewer import application
# from dm_control  import mjcf

# def main():
#     ASSETS_DIR = os.path.join(os.path.dirname(__file__),'a1')
#     _A1_XML_PATH = os.path.join(ASSETS_DIR,'xml','a1.xml')
#     model = mjcf.from_path(_A1_XML_PATH)
#     env = suite.load(domain_name="locomotion",task_name="run")
#     viewer = application.Application()
#     viewer.launch(env)

#     if __name__ == "__main__":
#         main()


# import numpy as np
# from dm_control import mjcf
# from dm_control import suite
# from dm_control.viewer import viewer

# # Загрузка модели A1 из XML-файла
# mjcf_model = mjcf.from_path('a1/xml/a1.xml')  # Укажите путь к вашему XML-файлу

# # Создание физического окружения
# physics = mjcf.Physics.from_mjcf_model(mjcf_model)

# # Функция для инициализации визуализации
# def initialize_viewer():
#     view = viewer.launch(physics, title='A1 Robot Simulation')
#     return view

# # Запуск визуализации
# view = initialize_viewer()

# # Пример обновления физики (например, для приведения модели в начальное положение)
# with physics.reset_context():
#     physics.data.qpos[:] = np.asarray([0.05, 0.7, -1.4] * 4)  # Начальная конфигурация A1

# # Пример основного цикла симуляции
# try:
#     while True:
#         physics.step()  # Обновление шага симуляции
#         view.render()   # Отрисовка текущего состояния модели
# except KeyboardInterrupt:
#     view.close()

# import os
# import mujoco.viewer
# import time
# import numpy as np
# from dm_control import mjcf
# from dm_control import viewer
# import mujoco

# # Укажите путь к XML-файлу модели A1
# # Загрузка модели A1 из XML-файла

# mjcf_model = mujoco.MjModel.from_xml_path('a1/xml/a1.xml')
# #mjcf_model = mjcf.from_path('a1/xml/a1.xml')  # Укажите путь к вашему XML-файлу
# data = mujoco.MjData(mjcf_model)

# viewer = mujoco.viewer.launch(mjcf_model,data) 

# for _ in range(1):
#     if viewer.is_alive:
#         mujoco.mj_step(mjcf_model,data)
#         viewer.render()
#     else:
#         break

# viewer.close()


import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('unitree_go2/scene.xml')
data = mujoco.MjData(model)

# create the viewer object
#viewer = viewer = mujoco.viewer.launch_passive(model,data) 

# Найдем id шарнира, к которому будем применять воздействие
motor_name = "FL_hip"
motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)

# Основной цикл симуляции
#print(data.ctrl)
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    data.ctrl[motor_id] = np.random.uniform(-33.5, 33.5)
    mujoco.mj_step(model, data)
    viewer.sync()
# Закрываем viewer


# # Создание физического окружения MuJoCo
# physics = mjcf.Physics.from_mjcf_model(mjcf_model)

# # Функция обновления симуляции
# def simulation_step(physics):
#     physics.step()

# # Настройка начального положения
# with physics.reset_context():
#     physics.data.qpos[:] = np.asarray([0.05, 0.7, -1.4] * 4)

# # Запуск визуализации с использованием viewer.launch
# viewer.launch(physics, step_callable=simulation_step, title="A1 Robot Simulation")