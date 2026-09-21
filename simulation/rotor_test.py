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

gravity = 9.81


# --------------------------------------------------
# 3. Calculate hover thrust
# --------------------------------------------------

hover_thrust = mass * gravity
thrust_per_rotor = hover_thrust / 4.0


print("Drone mass =", mass, "kg")
print("Drone inertia =", inertia, "kg m^2")
print("Hover thrust =", hover_thrust, "N")
print("Thrust per rotor =", thrust_per_rotor, "N")


# --------------------------------------------------
# 4. Roll experiment
# --------------------------------------------------
#
# Front = Rotor 1
# Back  = Rotor 2
# Left  = Rotor 3
# Right = Rotor 4
#
# Total thrust remains 9.81 N.
#
# Only left/right thrusts are different,
# so we intentionally create roll torque.
# --------------------------------------------------

thrusts = [
    2.4525,   # Rotor 1 - Front
    2.4525,   # Rotor 2 - Back
    2.5025,   # Rotor 3 - Left
    2.4025    # Rotor 4 - Right
]


# --------------------------------------------------
# 5. Calculate expected roll torque
# --------------------------------------------------

arm_length = 0.45

roll_torque = arm_length * (thrusts[2] - thrusts[3])

print("Applied roll torque =", roll_torque, "N m")


# --------------------------------------------------
# 6. Run simulation
# --------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    next_print = 0.5

    while viewer.is_running() and data.time < 5.0:

        # Apply rotor forces and torques
        apply_rotor_forces(model, data, thrusts)

        # Advance MuJoCo simulation
        mujoco.mj_step(model, data)


        # --------------------------------------------------
        # Print state every 0.5 seconds
        # --------------------------------------------------

        if data.time >= next_print:

            # MuJoCo free-joint quaternion:
            # [w, x, y, z]

            quat = data.qpos[3:7]

            w, x, y, z = quat


            # Roll
            roll = np.degrees(
                np.arctan2(
                    2 * (w * x + y * z),
                    1 - 2 * (x * x + y * y)
                )
            )


            # Pitch
            pitch = np.degrees(
                np.arcsin(
                    np.clip(
                        2 * (w * y - z * x),
                        -1.0,
                        1.0
                    )
                )
            )


            # Yaw
            yaw = np.degrees(
                np.arctan2(
                    2 * (w * z + x * y),
                    1 - 2 * (y * y + z * z)
                )
            )


            # Print simulation state
            print(
                f"Time = {data.time:.1f} s | "
                f"Z = {data.qpos[2]:.3f} m | "
                f"Roll = {roll:.2f} deg | "
                f"Pitch = {pitch:.2f} deg | "
                f"Yaw = {yaw:.2f} deg"
            )


            next_print += 0.5


        # Update MuJoCo viewer
        viewer.sync()