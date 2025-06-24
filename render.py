from stable_baselines3 import PPO
import gym
import numpy as np
from gym import spaces
import mujoco
from mujoco import MjModel, MjData, viewer
import time

class FurutaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, seed=0, render=False):
        super().__init__()
        self.model = MjModel.from_xml_path("scene.xml")
        self.data = MjData(self.model)
        self.render_mode = render
        self.viewer = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        high = np.array([2*np.pi, 1000, 2*np.pi, 300], dtype=np.float32)
        low = np.array([0, -1000, 0, -300], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.data.ctrl[0] = action[0]*300
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = False
        return obs, reward, done, {}, {}

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        theta = self.data.qpos[1] % (2 * np.pi)
        theta_dot = self.data.qvel[1]
        phi = self.data.qpos[0] % (2 * np.pi)
        phi_dot = self.data.qvel[0]
        return np.array([theta, theta_dot, phi, phi_dot], dtype=np.float32)

    def _compute_reward(self, obs):
        theta, theta_dot, _, phi_dot = obs
        if np.pi*0.9 <= theta <= np.pi:
            reward = np.cos(theta) + (1 + np.cos(theta)) ** 3 - 0.1*(theta_dot**2 + phi_dot**2)
        else:
            reward = -0.2
        return reward

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()  # <-- this line updates the viewer every frame


env = FurutaEnv(render=True)
model = PPO.load("ppo_furuta")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.004)
