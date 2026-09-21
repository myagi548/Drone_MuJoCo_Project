import numpy as np
import mujoco
import mujoco.viewer

from rotor_model import apply_rotor_forces


# --------------------------------------------------
# 1. Load MuJoCo model
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path("drone_model.xml")
data = mujoco.MjData(model)


# --------------------------------------------------
# 2. Get drone properties
# --------------------------------------------------

drone_id = model.body("drone").id

mass = model.body_mass[drone_id]
inertia = model.body_inertia[drone_id]

Ixx = inertia[0]

gravity = 9.81


# --------------------------------------------------
# 3. Calculate hover thrust
# --------------------------------------------------

hover_thrust = mass * gravity
hover_per_rotor = hover_thrust / 4.0


# --------------------------------------------------
# 4. Define a SMALL roll torque
# --------------------------------------------------

# Rotor positions:
# Rotor 1 = Front
# Rotor 2 = Back
# Rotor 3 = Left
# Rotor 4 = Right

arm_length = 0.45

# Small thrust difference between left and right
delta_thrust = 0.01

left_thrust = hover_per_rotor + delta_thrust
right_thrust = hover_per_rotor - delta_thrust

thrusts_hover = [
    hover_per_rotor,
    hover_per_rotor,
    hover_per_rotor,
    hover_per_rotor
]

thrusts_roll = [
    hover_per_rotor,
    hover_per_rotor,
    left_thrust,
    right_thrust
]


# --------------------------------------------------
# 5. Calculate applied roll torque
# --------------------------------------------------

roll_torque = arm_length * (
    left_thrust - right_thrust
)


# --------------------------------------------------
# 6. Calculate expected angular acceleration
# --------------------------------------------------

expected_angular_acceleration = roll_torque / Ixx


# --------------------------------------------------
# 7. Print experiment information
# --------------------------------------------------

print("==============================================")
print("ROLL RESPONSE EXPERIMENT")
print("==============================================")

print(f"Drone mass = {mass:.6f} kg")

print(
    f"Drone inertia = "
    f"[{inertia[0]:.8f}, "
    f"{inertia[1]:.8f}, "
    f"{inertia[2]:.8f}] kg m^2"
)

print(f"Ixx = {Ixx:.8f} kg m^2")

print(f"Hover thrust = {hover_thrust:.6f} N")

print(f"Hover thrust per rotor = {hover_per_rotor:.6f} N")

print(f"Left rotor thrust = {left_thrust:.6f} N")

print(f"Right rotor thrust = {right_thrust:.6f} N")

print(f"Applied roll torque = {roll_torque:.6f} N m")

print(
    f"Expected angular acceleration = "
    f"{expected_angular_acceleration:.6f} rad/s^2"
)

print("==============================================")


# --------------------------------------------------
# 8. Simulation
# --------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    next_print = 0.1

    while viewer.is_running() and data.time < 2.0:

        # --------------------------------------------------
        # Apply roll torque only for first 0.2 seconds
        # --------------------------------------------------

        if data.time < 0.2:

            apply_rotor_forces(
                model,
                data,
                thrusts_roll
            )

        else:

            # After 0.2 seconds return to equal hover thrust
            apply_rotor_forces(
                model,
                data,
                thrusts_hover
            )


        # Advance simulation
        mujoco.mj_step(model, data)


        # --------------------------------------------------
        # Calculate roll angle
        # --------------------------------------------------

        if data.time >= next_print:

            quat = data.qpos[3:7]

            w, x, y, z = quat


            roll = np.degrees(
                np.arctan2(
                    2 * (w * x + y * z),
                    1 - 2 * (x * x + y * y)
                )
            )


            # --------------------------------------------------
            # Get angular velocity
            #
            # MuJoCo stores angular velocity in qvel[3:6]
            # for a free joint.
            # --------------------------------------------------

            angular_velocity = data.qvel[3:6]

            roll_rate = np.degrees(
                angular_velocity[0]
            )


            print(
                f"Time = {data.time:.2f} s | "
                f"Z = {data.qpos[2]:.3f} m | "
                f"Roll = {roll:.3f} deg | "
                f"Roll rate = {roll_rate:.3f} deg/s"
            )


            next_print += 0.1


        viewer.sync()


print("==============================================")
print("Experiment finished.")
print("==============================================")