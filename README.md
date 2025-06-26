# Reinforcement-Learning-with-Mujoco
A repository for studies with Mujoco and reinforcement learning frameworks
Here you will find a case study with furuta pendulum with stable baselines 3

Requirements: gymnasium, stable_baselines_3, mujoco

* To train Furuta pendulum, use ```furuta_env.py```
* To evaluate trained model, just run ```render.py``` with your trained model
* To check ideal scenario, just run ```control_furuta.py``` to check ideal model behavior (PID tuned model)
* To watch performances in browser locally, pip install tensorboard then ```tensorboard --logdir ./furuta_tensorboard/```
* If you are using Windows, just download WSL then install the requirements inside virtual machine to run simulation.
