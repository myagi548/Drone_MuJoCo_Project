import numpy as np
import mujoco


# ==========================================================
# 1. PARAMETERS
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

mass = 1.0

gravity = 9.81

Ixx = 0.02296667

Kp = 4.04213392
Ki = 7.34933440
Kd = 0.64306676

dt = 0.002

initial_angle_deg = 2.0

initial_angle_rad = np.radians(
    initial_angle_deg
)


# ==========================================================
# 2. HOVER FORCE
# ==========================================================
#
# The drone must remain airborne during the roll test.
#
# F_hover = m*g
#

hover_force = (
    mass * gravity
)


# ==========================================================
# 3. THEORETICAL INITIAL CONDITION
# ==========================================================

initial_rate = 0.0

initial_acceleration = (
    -Kp
    / Ixx
    * initial_angle_rad
)


# ==========================================================
# 4. THEORETICAL RESPONSE
# ==========================================================
#
# Closed-loop characteristic equation:
#
# Ixx*s^3 + Kd*s^2 + Kp*s + Ki = 0
#
# Designed poles:
#
# -4, -4, -20
#
# Therefore:
#
# phi(t) =
#
# A*exp(-4t)
# + B*t*exp(-4t)
# + C*exp(-20t)
#

matrix = np.array(
    [
        [1.0, 0.0, 1.0],
        [-4.0, 1.0, -20.0],
        [16.0, -8.0, 400.0]
    ]
)

rhs = np.array(
    [
        initial_angle_rad,
        initial_rate,
        initial_acceleration
    ]
)

A, B, C = np.linalg.solve(
    matrix,
    rhs
)


# ==========================================================
# 5. THEORETICAL TIME
# ==========================================================

time = np.arange(
    0.0,
    5.0 + dt,
    dt
)


theory_rad = (
    A * np.exp(-4.0 * time)
    + B * time * np.exp(-4.0 * time)
    + C * np.exp(-20.0 * time)
)

theory_deg = np.degrees(
    theory_rad
)


# ==========================================================
# 6. THEORETICAL PERFORMANCE
# ==========================================================

theory_overshoot_angle = abs(
    np.min(theory_deg)
)

theory_overshoot_percent = (
    theory_overshoot_angle
    / initial_angle_deg
    * 100.0
)


theory_band = (
    0.02
    * initial_angle_deg
)


theory_settling_time = None

for i in range(len(time)):

    if np.all(
        np.abs(theory_deg[i:])
        <= theory_band
    ):

        theory_settling_time = time[i]

        break


# ==========================================================
# 7. LOAD MUJOCO
# ==========================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


# ==========================================================
# 8. INITIAL DRONE STATE
# ==========================================================

data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 1.0


# Quaternion:
#
# q = [cos(phi/2), sin(phi/2), 0, 0]
#

data.qpos[3] = np.cos(
    initial_angle_rad / 2.0
)

data.qpos[4] = np.sin(
    initial_angle_rad / 2.0
)

data.qpos[5] = 0.0
data.qpos[6] = 0.0


# Initial linear velocity

data.qvel[0:3] = 0.0

# Initial angular velocity

data.qvel[3:6] = 0.0


mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# 9. PID STATE
# ==========================================================

integral = 0.0


mujoco_time = []

mujoco_roll = []

mujoco_rate = []

mujoco_torque = []


drone_id = model.body(
    "drone"
).id


# ==========================================================
# 10. MUJOCO SIMULATION
# ==========================================================

simulation_time = 5.0

steps = int(
    simulation_time / dt
)


for step in range(steps):

    current_time = (
        step * dt
    )


    # ------------------------------------------------------
    # Roll angle from quaternion
    # ------------------------------------------------------

    quaternion = data.qpos[3:7]

    w, x, y, z = quaternion

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y)
    )


    # ------------------------------------------------------
    # Roll angular velocity
    # ------------------------------------------------------

    roll_rate = data.qvel[3]


    # ------------------------------------------------------
    # PID
    # ------------------------------------------------------

    desired_roll = 0.0

    error = (
        desired_roll
        - roll
    )


    integral += (
        error * dt
    )


    torque = (
        Kp * error
        + Ki * integral
        - Kd * roll_rate
    )


    # ------------------------------------------------------
    # Apply hover force
    # ------------------------------------------------------
    #
    # Upward force:
    #
    # Fz = m*g = 9.81 N
    #
    # This prevents the drone from falling during
    # the roll-controller validation.
    #

    data.xfrc_applied[
        drone_id,
        0
    ] = 0.0

    data.xfrc_applied[
        drone_id,
        1
    ] = 0.0

    data.xfrc_applied[
        drone_id,
        2
    ] = hover_force


    # ------------------------------------------------------
    # Apply roll torque
    # ------------------------------------------------------

    data.xfrc_applied[
        drone_id,
        3
    ] = torque

    data.xfrc_applied[
        drone_id,
        4
    ] = 0.0

    data.xfrc_applied[
        drone_id,
        5
    ] = 0.0


    # ------------------------------------------------------
    # MuJoCo step
    # ------------------------------------------------------

    mujoco.mj_step(
        model,
        data
    )


    # ------------------------------------------------------
    # Store data
    # ------------------------------------------------------

    mujoco_time.append(
        current_time
    )

    mujoco_roll.append(
        np.degrees(roll)
    )

    mujoco_rate.append(
        np.degrees(roll_rate)
    )

    mujoco_torque.append(
        torque
    )


# ==========================================================
# 11. CONVERT TO NUMPY ARRAYS
# ==========================================================

mujoco_time = np.array(
    mujoco_time
)

mujoco_roll = np.array(
    mujoco_roll
)

mujoco_rate = np.array(
    mujoco_rate
)

mujoco_torque = np.array(
    mujoco_torque
)


# ==========================================================
# 12. MUJOCO PERFORMANCE
# ==========================================================

mujoco_overshoot_angle = abs(
    np.min(mujoco_roll)
)

mujoco_overshoot_percent = (
    mujoco_overshoot_angle
    / initial_angle_deg
    * 100.0
)


mujoco_band = (
    0.02
    * initial_angle_deg
)


mujoco_settling_time = None


for i in range(
    len(mujoco_time)
):

    if np.all(
        np.abs(mujoco_roll[i:])
        <= mujoco_band
    ):

        mujoco_settling_time = (
            mujoco_time[i]
        )

        break


mujoco_sse = abs(
    mujoco_roll[-1]
)


max_rate = np.max(
    np.abs(mujoco_rate)
)

max_torque = np.max(
    np.abs(mujoco_torque)
)


# ==========================================================
# 13. THEORY vs MUJOCO
# ==========================================================

settling_difference = abs(
    theory_settling_time
    - mujoco_settling_time
)


overshoot_difference = abs(
    theory_overshoot_percent
    - mujoco_overshoot_percent
)


settling_difference_percent = (
    settling_difference
    / theory_settling_time
    * 100.0
)


overshoot_difference_percent = (
    overshoot_difference
    / theory_overshoot_percent
    * 100.0
)


# ==========================================================
# 14. DISPLAY
# ==========================================================

print()

print("==================================================")
print("PID THEORY vs MUJOCO VALIDATION")
print("==================================================")


print()

print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass = {mass:.4f} kg"
)

print(
    f"Gravity = {gravity:.4f} m/s^2"
)

print(
    f"Hover force = {hover_force:.4f} N"
)

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)


print()

print("CONTROLLER")
print("----------------------------------------------")

print(
    f"Kp = {Kp:.8f}"
)

print(
    f"Ki = {Ki:.8f}"
)

print(
    f"Kd = {Kd:.8f}"
)


print()

print("INITIAL CONDITION")
print("----------------------------------------------")

print(
    f"Initial roll = "
    f"{initial_angle_deg:.4f} deg"
)

print(
    "Initial roll rate = "
    "0.0000 deg/s"
)


print()

print("THEORETICAL RESULTS")
print("----------------------------------------------")

print(
    f"Settling time = "
    f"{theory_settling_time:.4f} s"
)

print(
    f"Overshoot = "
    f"{theory_overshoot_percent:.4f} %"
)

print(
    f"Overshoot angle = "
    f"{theory_overshoot_angle:.6f} deg"
)


print()

print("MUJOCO RESULTS")
print("----------------------------------------------")

print(
    f"Settling time = "
    f"{mujoco_settling_time:.4f} s"
)

print(
    f"Overshoot = "
    f"{mujoco_overshoot_percent:.4f} %"
)

print(
    f"Overshoot angle = "
    f"{mujoco_overshoot_angle:.6f} deg"
)

print(
    f"Steady-state error = "
    f"{mujoco_sse:.8f} deg"
)

print(
    f"Maximum roll rate = "
    f"{max_rate:.6f} deg/s"
)

print(
    f"Maximum torque = "
    f"{max_torque:.6f} N m"
)


print()

print("THEORY vs MUJOCO DIFFERENCE")
print("----------------------------------------------")

print(
    f"Settling-time difference = "
    f"{settling_difference:.6f} s"
)

print(
    f"Settling-time difference = "
    f"{settling_difference_percent:.4f} %"
)

print(
    f"Overshoot difference = "
    f"{overshoot_difference:.6f} %"
)

print(
    f"Overshoot difference = "
    f"{overshoot_difference_percent:.4f} %"
)


print()

print("==================================================")