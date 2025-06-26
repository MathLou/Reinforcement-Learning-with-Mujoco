import mujoco
import mujoco.viewer
from time import sleep
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Desired joint positions (in radians)
i = 0
positions = [-90, 90]  # Example positions in degrees
error_integral = 0.0  # Initialize integral term for PID control
speeds = [-70,0,70,0]
oscilation = [-35, -3,-3, 35, 5,5]  # Oscillation values for control signal
oscilate = False  # Flag to indicate oscillation
const = 0
# Launch viewer to visualize
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        #target_coxa = np.deg2rad(positions[j])   
        # Read current positions
        q_coxa = data.qpos[model.joint("arm").qposadr]
        angular_speed = data.qvel[0]
        # Calculating angles
        angle_read = (data.qpos[1]*180/np.pi)%360
        const += 0.0001/(180-angle_read) # Constant based on the angle
        movable_setpoint = 180 - const
        error = movable_setpoint - angle_read
        if i%100 == 0:
            print(f'{180 - round(angle_read,2)}, speed: {round(angular_speed*30/np.pi,2)}')  # Print the position in degrees
        # Step simulation
        mujoco.mj_step(model, data)
        sleep(0.0015)
        # PID for stabilization
        if not oscilate and i%5:
            kp = 0.03
            ki = 0.0025
            kd = 0.0002
            error_integral += error  # Integral term
            # anti windup
            threshold = 300
            if error_integral > threshold:
                error_integral = threshold
            elif error_integral < -threshold:
                error_integral = -threshold
            # Calculate control signal
            u = (kp* error + ki * error_integral + kd * (error - data.ctrl[0]) / 0.001)
            target_speed_rpm = u
            target_speed = target_speed_rpm*np.pi/30  # Convert RPM to rad/s
            data.ctrl[0] = target_speed
        angle_threshold = 90  # Threshold for oscillation detection
        if np.abs(error) >= angle_threshold:
            oscilate = True
            error_integral = 0.0  # Reset integral term when oscillating
            u = 0  # Reset control signal when oscillating
        elif np.abs(error) < angle_threshold:
            oscilate = False
        if oscilate and i%100 == 0:
            data.ctrl[0] = oscilation[i%len(oscilation)]
        # Update control signal
        i += 1
        viewer.sync()
