"""
Setup:
pip install mujoco numpy
"""

import numpy as np
import ppo.ppo_agent as PPO

class InvertedPendulumEnv:
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.ball_position = [0,0,0]
        self.reset_model()

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        reward = 1.0
        ob = self.obs()
        terminated = bool(not np.isfinite(ob).all())
        if terminated:
            print(ob)
        return ob, reward, terminated

    def obs(self):
        pendulum_state = np.concatenate([self.data.qpos, self.data.qvel])  # форма (4,)
        ball_x = np.array([self.ball_position[0]])  # Берём только X-координату
        
        return np.concatenate([pendulum_state, ball_x])  # форма (5,)
    
    def reset_model(self):
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        self.data.qpos[1] = 3.14  # Set the pole to be facing down
        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position, color=[1, 0, 0, 1], radius=0.01):
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.array(position),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        self.viewer.user_scn.ngeom = 1
        self.ball_position = position

    @property
    def current_time(self):
        return self.data.time


import time
import mujoco
import mujoco.viewer

env = InvertedPendulumEnv()
#print(ob.size)#print(ob[3]) # [2] - скорость слайда 3 - скорость колонны 
agent = PPO.PPOAgent(env, lr=3e-4, hidden_dim=128, load_path='good_dynamic_weights/ppo_8000_7084.pth')
#agent.train_ppo(env, agent, episodes=150000, dynamic=True)

"""ОРИГИНАЛЬНЫЙ ПАЙПЛАЙН"""
last_update = 0
action = 0
while env.current_time < 5000:
    if env.current_time - last_update > 5:
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        env.draw_ball(target_pos, radius=0.05)
        last_update = env.current_time
    ob, reward, terminated = env.step(action)
    action, _ =agent.act(ob)
    time.sleep(0.01)

#angle = next_state[1]
##if angle > np.pi:
##    angle = angle - 2*np.pi
#reward = np.cos(angle)  # Max reward when angle=0