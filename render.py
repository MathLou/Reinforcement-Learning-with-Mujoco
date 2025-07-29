from stable_baselines3 import PPO, SAC
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData, viewer
import time
from furuta_env import FurutaEnv 

# good results: ppo_furuta_10M_PPO41,sac_furuta_10M_43

env = FurutaEnv()
model = SAC.load("sac_furuta_10M_43", env=env)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Reward: {reward}, Observation {obs}, done: {done}")
    time.sleep(0.008)
