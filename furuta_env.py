import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import mujoco
from mujoco import MjModel, MjData
from time import sleep
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
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
        high = np.array([2*np.pi, np.inf, np.inf, np.inf], dtype=np.float32)
        low = np.array([0, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def step(self, action): # for SAC
        self.data.ctrl[0] = action[0] * 300
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)

        # Custom termination if phi exceeds bounds
        terminated = abs(obs[2]) > np.pi / 2  # φ out-of-bounds
        truncated = False  # Let TimeLimit decide if truncated

        info = {}
        return obs, reward, terminated, truncated, info
    # def step(self, action):  for PPO
    #     max_velocity = 300
    #     self.data.ctrl[0] = action[0]*max_velocity 
    #     mujoco.mj_step(self.model, self.data)
    #     obs = self._get_obs()
    #     theta, theta_dot, phi, phi_dot = obs
    #     reward = self._compute_reward(obs)
    #     # End episode if phi goes too far
    #     # if abs(phi) > np.pi / 2:
    #     #     done = True
    #     # else:
    #     #     done = False
    #     done = False
    #     return obs, reward, done, {}, {}

    def reset(self, *, seed=42, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.01 * np.random.randn(2)  # small random angles
        self.data.qvel[:] = 0.01 * np.random.randn(2)  # small random velocities
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        theta = np.rad2deg(self.data.qpos[1])%360  # Convert to degrees and wrap around
        theta = np.deg2rad(theta)  # Convert back to radians for consistency
        theta_dot = self.data.qvel[1]
        phi = np.rad2deg(self.data.qpos[0]) % 360
        phi = np.deg2rad(phi)
        phi_dot = self.data.qvel[0]
        return np.array([theta, theta_dot, phi, phi_dot], dtype=np.float32)

    def _compute_reward(self, obs):
        theta, theta_dot, phi, phi_dot = obs
        # angle_offset = 0.62 #10M experimento --> furuta 10m and 2m
        #angle_offset = 0.62  # 2M experiment --> furuta 2m e ppo15
        #angle_offset = 0.62  # 2M experiment V3--> furuta 2m, 10m (pp0_9) (pp0_11) (ppo_12) (ppo_13)
        #angle_offset = 0.17 # 100k v5 
        angle_offset = 0.62  # ppo 200 milhoes
        if (np.pi - angle_offset) < theta and theta < (np.pi +angle_offset):
            #reward = 7 - np.sin(theta) - (1 + np.sin(theta)) ** 3 #todas as versões abaixo de ppo10, mas tbm inclui ppo_11
            #reward = -7 + np.sin(theta) + (1 + np.sin(theta)) ** 3 + (1/theta_dot + 1/phi_dot) # experimento pp10
            #reward = 1/np.abs(theta - np.pi) #ppo13
            #reward = 7 - np.cos(theta) - (1 + np.cos(theta)) ** 3 - 0.01*(theta_dot*0.0001 + phi_dot/300) #ppo14
            #reward = 7 - np.cos(theta) - (1 + np.cos(theta)) ** 3 + - 0.5*np.abs(np.sin(phi)) - (np.abs(phi_dot))/80 #ppo17 with angle_offset = 0.3
            #reward = 7 - np.cos(theta) - (1 + np.cos(theta)) ** 3 + - (np.abs(phi_dot))/300 # ppo19, angle offset = 0.3
            ohmega  = 3 ##ppo41 ohmegas = 5
            reward = 7 - np.cos(ohmega*theta) - (1 + np.cos(ohmega*theta))**3 # sacs
            #reward = 5 - np.cos(ohmega*theta) - (1 + np.cos(ohmega*theta))**3 -3*np.abs(np.sin(phi)) - 5*(np.abs(phi_dot/300)) -0.01*(np.abs(theta_dot)) # ppo 200 milhoes
            #reward = np.exp(-np.abs(theta - np.pi)) - 0.003*(np.abs(theta_dot)) - 0.003*(np.abs(phi_dot)) # ppo 41
            #reward = 5 - np.cos(ohmega*theta) - (1 + np.cos(ohmega*theta))**3 -3*np.abs(np.sin(phi)) - 2*(np.abs(phi_dot/300)) -0.001*(np.abs(theta_dot)) ##ppo39
            #reward =  - np.cos(theta) - (1 + np.cos(theta)) ** 3 #ppo15, ppo16
        else:
            #reward = -0.001 ## ppo 35
            reward = -0.2 ## ppo 36,38 (0,2) ppo 37 (reward = -1) and 39
        #reward -=  0.5*(np.abs(phi_dot))/(70) # experimento v4 e v5
        #reward -=  0.015*(np.abs(phi_dot))/(300) # experimento v6 e ppo_9 
        return reward

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

if __name__ == "__main__":
    #time_steps = 2_000_000 # 36
    time_steps = 10_000_000 # 37
    # ppo 35 was 20 m
    experiment_number = 43  # Change this number for different experiments
    env = Monitor(TimeLimit(FurutaEnv(), max_episode_steps=800)) # all before 39 was 500

    model = SAC("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log="./furuta_tensorboard/")
    model.learn(total_timesteps=time_steps)
    
    # if time_steps == 1_000_000:
    #     model.save(f"ppo_furuta_1M_PPO{experiment_number}")
    # elif time_steps == 500_000:
    #     model.save(f"ppo_furuta_500k_PPO{experiment_number}")
    # elif time_steps == 2_000_000:
    #     model.save(f"ppo_furuta_2M_PPO{experiment_number}")
    # elif time_steps == 4_000_000:
    #     model.save(f"ppo_furuta_4M_PPO{experiment_number}")
    # elif time_steps == 10_000_000:
    #     model.save(f"ppo_furuta_10M_PPO{experiment_number}")
    # elif time_steps == 100_000:
    #     model.save(f"ppo_furuta_100k_PPO{experiment_number}")
    # elif time_steps == 200_000_000:
    #     model.save(f"ppo_furuta_200M_PPO{experiment_number}")
    model.save(f"sac_furuta_10M_{experiment_number}")  #sac 42 -> (ohmega =5,offset = 4,reward =  5), 43 ->(ohmega = 3, offset = 0.62, reward = 7)
  
