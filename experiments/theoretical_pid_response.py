import numpy as np


# ==========================================================
# 1. ACTUAL DRONE PARAMETERS
# ==========================================================

Ixx = 0.02296667

Kp = 4.04213392
Ki = 7.34933440
Kd = 0.64306676


# ==========================================================
# 2. INITIAL CONDITION
# ==========================================================

initial_angle_deg = 2.0

initial_angle = np.radians(
    initial_angle_deg
)

initial_rate = 0.0

# The PID integral starts from zero.
initial_integral = 0.0


# ==========================================================
# 3. CLOSED-LOOP CHARACTERISTIC EQUATION
# ==========================================================
#
# Ixx*s^3 + Kd*s^2 + Kp*s + Ki = 0
#
# With our gains:
#
# 0.02296667*s^3
# + 0.64306676*s^2
# + 4.04213392*s
# + 7.34933440 = 0
#
# The intended poles are:
#
# -4, -4, -20
#

poles = np.roots(
    [
        Ixx,
        Kd,
        Kp,
        Ki
    ]
)


# ==========================================================
# 4. INITIAL ACCELERATION
# ==========================================================
#
# Actual PID:
#
# torque =
#     Kp*error
#     + Ki*integral
#     - Kd*angular_rate
#
# Desired angle = 0
#
# Therefore:
#
# error = -phi
#
# At t = 0:
#
# integral = 0
# angular_rate = 0
#
# Hence:
#
# torque(0) = -Kp*phi(0)
#
# Plant:
#
# Ixx*phi_ddot = torque
#
# Therefore:
#
# phi_ddot(0) =
#     -Kp/Ixx * phi(0)
#

initial_acceleration = (
    -Kp
    / Ixx
    * initial_angle
)


# ==========================================================
# 5. NATURAL RESPONSE
# ==========================================================
#
# Because the poles are approximately:
#
# -4, -4, -20
#
# the response has the form:
#
# phi(t) =
#
# A*exp(-4t)
# + B*t*exp(-4t)
# + C*exp(-20t)
#

matrix = np.array(
    [
        # phi(0)
        [1.0, 0.0, 1.0],

        # phi_dot(0)
        [-4.0, 1.0, -20.0],

        # phi_ddot(0)
        [16.0, -8.0, 400.0]
    ]
)

right_side = np.array(
    [
        initial_angle,
        initial_rate,
        initial_acceleration
    ]
)


A, B, C = np.linalg.solve(
    matrix,
    right_side
)


# ==========================================================
# 6. TIME
# ==========================================================

time = np.linspace(
    0.0,
    5.0,
    5001
)


# ==========================================================
# 7. THEORETICAL RESPONSE
# ==========================================================

response_rad = (
    A * np.exp(-4.0 * time)
    + B * time * np.exp(-4.0 * time)
    + C * np.exp(-20.0 * time)
)

response_deg = np.degrees(
    response_rad
)


# ==========================================================
# 8. OVERSHOOT
# ==========================================================
#
# Desired final angle = 0.
#
# Therefore any negative value is an overshoot
# beyond the target.
#

negative_response = response_deg[
    response_deg < 0.0
]


if len(negative_response) > 0:

    overshoot_angle = abs(
        np.min(negative_response)
    )

else:

    overshoot_angle = 0.0


overshoot_percent = (
    overshoot_angle
    / initial_angle_deg
    * 100.0
)


# ==========================================================
# 9. SETTLING TIME
# ==========================================================
#
# We use a 2% band around the desired value.
#
# For an initial angle of 2 degrees:
#
# 2% band = 0.04 degrees
#

settling_band = (
    0.02
    * initial_angle_deg
)


settling_time = None


for i in range(len(time)):

    remaining = np.abs(
        response_deg[i:]
    )

    if np.all(
        remaining <= settling_band
    ):

        settling_time = time[i]

        break


# ==========================================================
# 10. DISPLAY
# ==========================================================

print()

print("==============================================")
print("CORRECTED THEORETICAL PID RESPONSE")
print("==============================================")

print()

print("PID PARAMETERS")
print("----------------------------------------------")

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)

print(
    f"Kp  = {Kp:.8f}"
)

print(
    f"Ki  = {Ki:.8f}"
)

print(
    f"Kd  = {Kd:.8f}"
)


print()

print("CALCULATED POLES")
print("----------------------------------------------")

for pole in poles:

    print(
        f"{pole.real:.8f}"
        f" {pole.imag:+.8f}j"
    )


print()

print("INITIAL CONDITIONS")
print("----------------------------------------------")

print(
    f"Initial angle = "
    f"{initial_angle_deg:.6f} deg"
)

print(
    f"Initial angular velocity = "
    f"{initial_rate:.6f} rad/s"
)

print(
    f"Initial PID integral = "
    f"{initial_integral:.6f}"
)


print()

print("INITIAL ACCELERATION")
print("----------------------------------------------")

print(
    f"phi_ddot(0) = "
    f"{initial_acceleration:.8f} rad/s^2"
)


print()

print("RESPONSE COEFFICIENTS")
print("----------------------------------------------")

print(
    f"A = {A:.10f}"
)

print(
    f"B = {B:.10f}"
)

print(
    f"C = {C:.10f}"
)


print()

print("THEORETICAL RESULTS")
print("----------------------------------------------")

if settling_time is not None:

    print(
        f"Settling time = "
        f"{settling_time:.4f} s"
    )

else:

    print(
        "Settling time = Not reached"
    )


print(
    f"Overshoot angle = "
    f"{overshoot_angle:.10f} deg"
)

print(
    f"Overshoot = "
    f"{overshoot_percent:.10f} %"
)


print()

print("==============================================")