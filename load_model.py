import mujoco
import mujoco.viewer
import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the MJCF file
xml_path = os.path.join(script_dir, "mbd/assets/zx120/zx120.xml")

# Load the model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Launch the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Define the target velocity for the wheels (in rad/s)
    target_velocity = 0.1

    # Get the actuator IDs for the wheels
    wheel_actuators = [
        # "left_middle_wheel_joint_ctrl",
        # "left_front_wheel_joint_ctrl",
        "left_rear_wheel_joint_ctrl",
        # "right_middle_wheel_joint_ctrl",
        # "right_front_wheel_joint_ctrl",
        "right_rear_wheel_joint_ctrl",
    ]
    actuator_ids = [m.actuator(name).id for name in wheel_actuators]

    while viewer.is_running():
        # Set the control signal for each wheel actuator
        for i in actuator_ids:
            d.ctrl[i] = target_velocity

        # Step the simulation
        mujoco.mj_step(m, d)

        # Render the scene
        viewer.sync()
