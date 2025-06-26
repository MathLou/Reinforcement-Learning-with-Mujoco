import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData
from time import sleep
from stable_baselines3 import PPO
import mujoco.viewer

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
        max_velocity = 70
        self.data.ctrl[0] = action[0]*max_velocity
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = False
        sleep(0.0015)
        return obs, reward, done, {}, {}

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        theta = np.rad2deg(self.data.qpos[1])%360  # Convert to degrees and wrap around
        theta = np.deg2rad(theta)  # Convert back to radians for consistency
        theta_dot = self.data.qvel[1]
        phi = np.rad2deg(self.data.qpos[0]) % 360
        phi = np.deg2rad(phi)  # Convert back to radians for consistency
        phi_dot = self.data.qvel[0]
        return np.array([theta, theta_dot, phi, phi_dot], dtype=np.float32)

    def _compute_reward(self, obs):
        theta, theta_dot, _, phi_dot = obs
        angle_offset = 0.3  #in radians
        if (np.pi - angle_offset) <= theta <= (np.pi +angle_offset):
            reward =  - np.cos(theta) - (1 + np.cos(theta)) ** 3
        else:
            reward = -0.2 
        return reward

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

time_steps = 2_000_000 # Choose between 1M,2M,4M or 10M
experiment_number = 1  # Change this number for different experiments
env = FurutaEnv()

model = PPO("MlpPolicy", env, verbose=1,device="cuda",tensorboard_log="./furuta_tensorboard/")

model.learn(total_timesteps=time_steps)
if time_steps == 1_000_000:
    model.save(f"ppo_furuta_1M_PPO{experiment_number}")
elif time_steps == 2_000_000:
    model.save(f"ppo_furuta_2M_PPO{experiment_number}")
elif time_steps == 4_000_000:
    model.save(f"ppo_furuta_4M_PPO{experiment_number}")
elif time_steps == 10_000_000:
    model.save(f"ppo_furuta_10M_PPO{experiment_number}")
