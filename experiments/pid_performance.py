import csv
import numpy as np
import mujoco

from simulation.rotor_model import apply_rotor_forces
from simulation.mixer import mix_roll_torque
from control.pid_controller import PIDController


# ==========================================================
# 1. LOAD MUJOCO MODEL
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
# 3. PID GAINS
# ==========================================================
#
# These gains came from the critical-damping design:
#
# zeta = 1.0
# settling-time target = 1.0 s
# third-pole factor = 5.0
#

kp = 4.04213392

ki = 7.34933440

kd = 0.64306676


dt = model.opt.timestep


pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt
)


# ==========================================================
# 4. INITIAL CONDITION
# ==========================================================

initial_roll_deg = 10.0

initial_roll_rad = np.radians(
    initial_roll_deg
)


# Quaternion for roll rotation:
#
# q = [cos(phi/2), sin(phi/2), 0, 0]
#

data.qpos[3] = np.cos(
    initial_roll_rad / 2.0
)

data.qpos[4] = np.sin(
    initial_roll_rad / 2.0
)

data.qpos[5] = 0.0

data.qpos[6] = 0.0


mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# 5. DESIRED ROLL
# ==========================================================

desired_roll_rad = 0.0


# ==========================================================
# 6. DATA STORAGE
# ==========================================================

times = []

roll_angles = []

roll_rates = []

torques = []

total_thrusts = []


# ==========================================================
# 7. SIMULATION
# ==========================================================

simulation_time = 5.0


while data.time < simulation_time:

    # ------------------------------------------------------
    # Read quaternion
    # ------------------------------------------------------

    quat = data.qpos[3:7]

    w, x, y, z = quat


    # ------------------------------------------------------
    # Calculate roll angle
    # ------------------------------------------------------

    roll_rad = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (
            x * x + y * y
        )
    )


    # ------------------------------------------------------
    # Roll angular velocity
    # ------------------------------------------------------

    roll_rate_rad = data.qvel[3]


    # ------------------------------------------------------
    # PID controller
    # ------------------------------------------------------

    roll_torque = pid.compute(
        desired_angle_rad=desired_roll_rad,
        actual_angle_rad=roll_rad,
        angular_rate_rad=roll_rate_rad
    )


    # ======================================================
    # VERTICAL THRUST COMPENSATION
    # ======================================================
    #
    # When the drone rolls by phi:
    #
    #       vertical thrust = T*cos(phi)
    #
    # We want:
    #
    #       T*cos(phi) = mg
    #
    # Therefore:
    #
    #       T = mg/cos(phi)
    #
    # ------------------------------------------------------

    cosine_roll = np.cos(
        roll_rad
    )


    # Safety check against division by a very small number.

    if cosine_roll < 0.1:

        cosine_roll = 0.1


    total_thrust = (
        mass
        * gravity
        / cosine_roll
    )


    # ------------------------------------------------------
    # Convert thrust + roll torque into rotor thrusts
    # ------------------------------------------------------

    thrusts = mix_roll_torque(
        total_thrust=total_thrust,
        roll_torque=roll_torque,
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
    # Record current state
    # ------------------------------------------------------

    times.append(
        data.time
    )

    roll_angles.append(
        np.degrees(roll_rad)
    )

    roll_rates.append(
        np.degrees(roll_rate_rad)
    )

    torques.append(
        roll_torque
    )

    total_thrusts.append(
        total_thrust
    )


    # ------------------------------------------------------
    # Advance MuJoCo simulation
    # ------------------------------------------------------

    mujoco.mj_step(
        model,
        data
    )


# ==========================================================
# 8. CONVERT DATA TO ARRAYS
# ==========================================================

times = np.array(times)

roll_angles = np.array(
    roll_angles
)

roll_rates = np.array(
    roll_rates
)

torques = np.array(
    torques
)

total_thrusts = np.array(
    total_thrusts
)


# ==========================================================
# 9. MAXIMUM ROLL
# ==========================================================

maximum_roll = np.max(
    np.abs(roll_angles)
)


# ==========================================================
# 10. OVERSHOOT
# ==========================================================

negative_rolls = roll_angles[
    roll_angles < 0.0
]


if len(negative_rolls) > 0:

    overshoot_angle = abs(
        np.min(negative_rolls)
    )

else:

    overshoot_angle = 0.0


overshoot_percent = (
    overshoot_angle
    / initial_roll_deg
    * 100.0
)


# ==========================================================
# 11. SETTLING TIME
# ==========================================================
#
# Initial error = 10 degrees.
#
# 2% of initial error:
#
#     0.02 * 10 = 0.2 degrees
#
# Settled means all later samples remain inside:
#
#     -0.2 <= roll <= +0.2
#

settling_band = (
    0.02
    * initial_roll_deg
)

settling_time = None


for i in range(
    len(times)
):

    remaining_roll = np.abs(
        roll_angles[i:]
    )

    if np.all(
        remaining_roll
        <= settling_band
    ):

        settling_time = times[i]

        break


# ==========================================================
# 12. STEADY-STATE ERROR
# ==========================================================

steady_state_error = abs(
    roll_angles[-1]
)


# ==========================================================
# 13. MAXIMUM ROLL RATE
# ==========================================================

maximum_roll_rate = np.max(
    np.abs(roll_rates)
)


# ==========================================================
# 14. MAXIMUM CONTROL TORQUE
# ==========================================================

maximum_torque = np.max(
    np.abs(torques)
)


# ==========================================================
# 15. MAXIMUM TOTAL THRUST
# ==========================================================

maximum_total_thrust = np.max(
    total_thrusts
)


# ==========================================================
# 16. SAVE RESULTS
# ==========================================================

output_file = (
    "results/pid_roll_response.csv"
)


with open(
    output_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "time",
            "roll_deg",
            "roll_rate_deg_s",
            "torque_Nm",
            "total_thrust_N"
        ]
    )

    for i in range(
        len(times)
    ):

        writer.writerow(
            [
                times[i],
                roll_angles[i],
                roll_rates[i],
                torques[i],
                total_thrusts[i]
            ]
        )


# ==========================================================
# 17. DISPLAY RESULTS
# ==========================================================

print()
print("==============================================")
print("PID PERFORMANCE ANALYSIS")
print("==============================================")


print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass = {mass:.6f} kg"
)

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)


print()
print("CONTROLLER DESIGN")
print("----------------------------------------------")

print(
    "Damping ratio = 1.00"
)

print(
    "Damping condition = Critical damping"
)

print(
    "Settling-time target = 1.00 s"
)


print()
print("MEASURED SIMULATION RESULTS")
print("----------------------------------------------")

if settling_time is not None:

    print(
        f"Actual settling time = "
        f"{settling_time:.4f} s"
    )

else:

    print(
        "Actual settling time = "
        "Not reached"
    )


print(
    f"Actual overshoot = "
    f"{overshoot_percent:.4f} %"
)

print(
    f"Steady-state error = "
    f"{steady_state_error:.6f} deg"
)

print(
    f"Maximum absolute roll = "
    f"{maximum_roll:.6f} deg"
)

print(
    f"Maximum roll rate = "
    f"{maximum_roll_rate:.6f} deg/s"
)

print(
    f"Maximum control torque = "
    f"{maximum_torque:.6f} N m"
)

print(
    f"Maximum total thrust = "
    f"{maximum_total_thrust:.6f} N"
)


print()
print(
    f"Response data saved to: "
    f"{output_file}"
)

print("==============================================")