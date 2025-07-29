import matplotlib.pyplot as plt
import numpy as np

def plot_reward(theta,angle_offset):
    if (np.pi - angle_offset) < theta and theta < (np.pi +angle_offset):
        ohmega = 3
        reward = 7 - np.cos(ohmega*theta) - (1 + np.cos(ohmega*theta))**3 #todas as versões abaixo de ppo10, mas tbm inclui ppo_11     
    else:
        reward = -0.2
    return reward


# Set the angle offset for the reward function
angle_offset = 0.62  # Adjust this value as needed
# Call the function to plot the reward function
theta_values = np.linspace(0, 2 * np.pi, 100)
rewards = [plot_reward(theta, angle_offset) for theta in theta_values]
plt.figure(figsize=(10, 5))
plt.plot(theta_values, rewards, label='Reward Function', color='blue')
plt.title('Reward Function for Furuta Pendulum')
plt.xlabel('Theta (radians)')
plt.ylabel('Reward')
plt.axhline(0, color='red', linestyle='--', label='Zero Reward Line')
plt.axvline(np.pi - angle_offset, color='green', linestyle='--', label='Lower Bound')
plt.axvline(np.pi + angle_offset, color='green', linestyle='--', label='Upper Bound')
plt.legend()
plt.grid()
plt.show()