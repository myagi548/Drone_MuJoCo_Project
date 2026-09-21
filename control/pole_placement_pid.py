
"""
Pole-placement PID controller for drone roll and pitch.

Model:
    I * theta_ddot = torque

Design:
    Desired polynomial:
    (s^2 + 2*zeta*wn*s + wn^2) * (s + p3)

The gains are calculated from the actual inertia supplied
by the MuJoCo model and the selected design parameters.
"""

class PolePlacementPID:
    def __init__(self, inertia, settling_time=1.0, zeta=1.0):
        # Physical parameter from MuJoCo
        self.inertia = float(inertia)

        # Design specifications
        self.settling_time = float(settling_time)
        self.zeta = float(zeta)

        # Third pole: chosen as 5 times the natural frequency
        self.wn = 4.0 / (self.zeta * self.settling_time)
        self.p3 = 5.0 * self.wn

        # Calculate gains by coefficient matching
        self.kd = self.inertia * (
            2.0 * self.zeta * self.wn + self.p3
        )

        self.kp = self.inertia * (
            self.wn**2
            + 2.0 * self.zeta * self.wn * self.p3
        )

        self.ki = self.inertia * (
            self.wn**2 * self.p3
        )

        # PID memory
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.first_step = True

    def update(self, desired_angle, actual_angle, dt):
        """
        Return the PID torque command.

        Angles must use the same unit (radians recommended).
        dt is the controller timestep in seconds.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive")

        error = desired_angle - actual_angle

        # Integral: sum error over time
        self.integral_error += error * dt

        # Derivative: rate of change of error
        if self.first_step:
            derivative_error = 0.0
            self.first_step = False
        else:
            derivative_error = (
                error - self.previous_error
            ) / dt

        self.previous_error = error

        # PID control law
        torque = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * derivative_error
        )

        return torque


def main():
    # Actual inertia values from your MuJoCo model
    IXX = 0.02296667
    IYY = 0.02296667

    # Simulation timestep
    DT = 0.002

    # Create independent roll and pitch controllers
    roll_pid = PolePlacementPID(
        inertia=IXX,
        settling_time=1.0,
        zeta=1.0
    )

    pitch_pid = PolePlacementPID(
        inertia=IYY,
        settling_time=1.0,
        zeta=1.0
    )

    print("\n--- Pole-Placement PID Design ---")
    print(f"Roll inertia:  {IXX:.8f} kg m^2")
    print(f"Pitch inertia: {IYY:.8f} kg m^2")
    print(f"Time step:     {DT:.4f} s")
    print(f"Settling time: {roll_pid.settling_time:.2f} s")
    print(f"Damping ratio: {roll_pid.zeta:.2f}")
    print(f"Natural freq.: {roll_pid.wn:.2f} rad/s")
    print(f"Third pole:    {roll_pid.p3:.2f} rad/s")

    print("\nCalculated gains:")
    print(
        f"Roll:  Kp={roll_pid.kp:.8f}, "
        f"Ki={roll_pid.ki:.8f}, "
        f"Kd={roll_pid.kd:.8f}"
    )
    print(
        f"Pitch: Kp={pitch_pid.kp:.8f}, "
        f"Ki={pitch_pid.ki:.8f}, "
        f"Kd={pitch_pid.kd:.8f}"
    )

    # Small example: correct a 10-degree roll error
    desired_roll = 0.0
    actual_roll = 10.0 * 3.141592653589793 / 180.0

    torque = roll_pid.update(
        desired_angle=desired_roll,
        actual_angle=actual_roll,
        dt=DT
    )

    print("\nExample:")
    print(f"Roll error: {desired_roll - actual_roll:.6f} rad")
    print(f"First-step PID torque: {torque:.6f} N m")


if __name__ == "__main__":
    main()