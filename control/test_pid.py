from pid_controller import PIDController



kp = 6.09636903
ki = 22.36141179
kd = 0.84933098

# MuJoCo timestep
dt = 0.002


pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt
)


desired_roll = 0.0
actual_roll = 10.0

import numpy as np

desired_roll_rad = np.radians(desired_roll)
actual_roll_rad = np.radians(actual_roll)

angular_rate = 0.0




torque = pid.compute(
    desired_angle_rad=desired_roll_rad,
    actual_angle_rad=actual_roll_rad,
    angular_rate_rad=angular_rate
)


print("==============================================")
print("PID CONTROLLER TEST")
print("==============================================")

print(f"Desired roll = {desired_roll:.2f} deg")
print(f"Actual roll  = {actual_roll:.2f} deg")

print(f"Desired roll = {desired_roll_rad:.6f} rad")
print(f"Actual roll  = {actual_roll_rad:.6f} rad")

print(f"Angle error  = {desired_roll_rad - actual_roll_rad:.6f} rad")

print(f"PID torque   = {torque:.6f} N m")

print("==============================================")