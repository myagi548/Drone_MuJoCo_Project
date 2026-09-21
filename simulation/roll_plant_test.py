import numpy as np
import mujoco

from simulation.rotor_model import apply_rotor_forces
from simulation.mixer import mix_roll_torque


# ==========================================================
# 1. LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path(
    "simulation/drone_model.xml"
)

data = mujoco.MjData(model)

drone_id = model.body("drone").id


# ==========================================================
# 2. PHYSICAL PARAMETERS
# ==========================================================

mass = model.body_mass[drone_id]

inertia = model.body_inertia[drone_id]

Ixx = inertia[0]

gravity = 9.81

arm_length = 0.45


# ==========================================================
# 3. EXPERIMENT SETTINGS
# ==========================================================

# Known roll torque applied to the drone.

test_torque = 0.01

simulation_time = 0.20

dt = model.opt.timestep


# ==========================================================
# 4. HOVER THRUST
# ==========================================================

# The drone is initially level.
#
# Therefore:
#
#       T = mg
#

total_thrust = mass * gravity


# ==========================================================
# 5. DATA STORAGE
# ==========================================================

times = []

roll_rates = []

roll_angles = []

angular_accelerations = []


# ==========================================================
# 6. INITIAL STATE
# ==========================================================

# Start level.

data.qpos[3] = 1.0
data.qpos[4] = 0.0
data.qpos[5] = 0.0
data.qpos[6] = 0.0

data.qvel[:] = 0.0


mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# 7. SIMULATION
# ==========================================================

previous_roll_rate = 0.0


while data.time < simulation_time:

    # ------------------------------------------------------
    # Read roll rate
    # ------------------------------------------------------

    roll_rate = data.qvel[3]


    # ------------------------------------------------------
    # Calculate roll angle
    # ------------------------------------------------------

    quat = data.qpos[3:7]

    w, x, y, z = quat

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (
            x * x + y * y
        )
    )


    # ------------------------------------------------------
    # Calculate angular acceleration
    #
    # alpha = change in angular velocity / time
    # ------------------------------------------------------

    angular_acceleration = (
        roll_rate - previous_roll_rate
    ) / dt


    # ------------------------------------------------------
    # Store measurements
    # ------------------------------------------------------

    times.append(data.time)

    roll_rates.append(
        np.degrees(roll_rate)
    )

    roll_angles.append(
        np.degrees(roll)
    )

    angular_accelerations.append(
        angular_acceleration
    )


    # ------------------------------------------------------
    # Convert desired torque into rotor thrusts
    # ------------------------------------------------------

    thrusts = mix_roll_torque(
        total_thrust=total_thrust,
        roll_torque=test_torque,
        arm_length=arm_length
    )


    # ------------------------------------------------------
    # Apply rotor forces
    # ------------------------------------------------------

    apply_rotor_forces(
        model,
        data,
        thrusts
    )


    # ------------------------------------------------------
    # Advance simulation
    # ------------------------------------------------------

    mujoco.mj_step(
        model,
        data
    )


    previous_roll_rate = roll_rate


# ==========================================================
# 8. CONVERT DATA
# ==========================================================

times = np.array(times)

roll_rates = np.array(roll_rates)

roll_angles = np.array(roll_angles)

angular_accelerations = np.array(
    angular_accelerations
)


# ==========================================================
# 9. REMOVE FIRST SAMPLE
# ==========================================================
#
# The first acceleration sample can be affected by the
# initial numerical condition.
#

valid_accelerations = (
    angular_accelerations[1:]
)


# ==========================================================
# 10. MEASURED ACCELERATION
# ==========================================================

measured_acceleration = np.mean(
    valid_accelerations
)


# ==========================================================
# 11. THEORETICAL ACCELERATION
# ==========================================================

theoretical_acceleration = (
    test_torque / Ixx
)


# ==========================================================
# 12. EXPERIMENTAL EFFECTIVE INERTIA
# ==========================================================

effective_inertia = (
    test_torque
    / measured_acceleration
)


# ==========================================================
# 13. ERROR
# ==========================================================

acceleration_error_percent = (
    abs(
        measured_acceleration
        - theoretical_acceleration
    )
    / abs(theoretical_acceleration)
    * 100.0
)


# ==========================================================
# 14. DISPLAY RESULTS
# ==========================================================

print()

print("==============================================")
print("ROLL PLANT EXPERIMENT")
print("==============================================")

print()

print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass = {mass:.6f} kg"
)

print(
    f"MuJoCo Ixx = {Ixx:.8f} kg m^2"
)

print(
    f"Test torque = {test_torque:.6f} N m"
)

print()

print("THEORETICAL MODEL")
print("----------------------------------------------")

print(
    "Equation: Ixx * alpha = torque"
)

print(
    f"Theoretical angular acceleration = "
    f"{theoretical_acceleration:.8f} rad/s^2"
)

print()

print("MEASURED SIMULATION")
print("----------------------------------------------")

print(
    f"Measured angular acceleration = "
    f"{measured_acceleration:.8f} rad/s^2"
)

print(
    f"Experimental effective inertia = "
    f"{effective_inertia:.8f} kg m^2"
)

print(
    f"Acceleration error = "
    f"{acceleration_error_percent:.4f} %"
)

print()

print("FINAL STATE")
print("----------------------------------------------")

print(
    f"Final roll angle = "
    f"{roll_angles[-1]:.6f} deg"
)

print(
    f"Final roll rate = "
    f"{roll_rates[-1]:.6f} deg/s"
)

print()

print("==============================================")