import numpy as np


class PIDController:
    

    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(
        self,
        desired_angle_rad,
        actual_angle_rad,
        angular_rate_rad
    ):
        

        error = (
            desired_angle_rad
            - actual_angle_rad
        )

        

        self.integral += error * self.dt

        
        torque = (
            self.kp * error
            + self.ki * self.integral
            - self.kd * angular_rate_rad
        )

        return torque