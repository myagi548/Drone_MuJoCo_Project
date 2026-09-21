
import numpy as np


class PositionController:
    """
    Converts position and velocity tracking errors
    into desired roll, pitch, and vertical acceleration.
    """

    def __init__(
        self,
        gravity,
        horizontal_gain,
        horizontal_damping,
        vertical_gain,
        vertical_damping,
        max_tilt_rad,
        max_horizontal_acceleration=2.5
    ):
        self.gravity = float(gravity)

        self.horizontal_gain = float(horizontal_gain)
        self.horizontal_damping = float(horizontal_damping)

        self.vertical_gain = float(vertical_gain)
        self.vertical_damping = float(vertical_damping)

        self.max_tilt_rad = float(max_tilt_rad)
        self.max_horizontal_acceleration = float(
            max_horizontal_acceleration
        )

    def compute(
        self,
        desired_position,
        actual_position,
        actual_velocity=None,
        desired_velocity=None
    ):
        desired_position = np.asarray(
            desired_position,
            dtype=float
        )

        actual_position = np.asarray(
            actual_position,
            dtype=float
        )

        if actual_velocity is None:
            actual_velocity = np.zeros(3)
        else:
            actual_velocity = np.asarray(
                actual_velocity,
                dtype=float
            )

        if desired_velocity is None:
            desired_velocity = np.zeros(3)
        else:
            desired_velocity = np.asarray(
                desired_velocity,
                dtype=float
            )

        position_error = (
            desired_position - actual_position
        )

        velocity_error = (
            desired_velocity - actual_velocity
        )

        # Horizontal acceleration command:
        # position correction + velocity tracking correction
        desired_x_acceleration = (
            self.horizontal_gain * position_error[0]
            + self.horizontal_damping * velocity_error[0]
        )

        desired_y_acceleration = (
            self.horizontal_gain * position_error[1]
            + self.horizontal_damping * velocity_error[1]
        )

        # Prevent excessive horizontal acceleration.
        desired_x_acceleration = np.clip(
            desired_x_acceleration,
            -self.max_horizontal_acceleration,
            self.max_horizontal_acceleration
        )

        desired_y_acceleration = np.clip(
            desired_y_acceleration,
            -self.max_horizontal_acceleration,
            self.max_horizontal_acceleration
        )

        # Convert acceleration commands to desired tilt.
        desired_pitch = (
            desired_x_acceleration / self.gravity
        )

        desired_roll = (
            -desired_y_acceleration / self.gravity
        )

        desired_roll = np.clip(
            desired_roll,
            -self.max_tilt_rad,
            self.max_tilt_rad
        )

        desired_pitch = np.clip(
            desired_pitch,
            -self.max_tilt_rad,
            self.max_tilt_rad
        )

        # Vertical position control.
        desired_z_acceleration = (
            self.vertical_gain * position_error[2]
            + self.vertical_damping * velocity_error[2]
        )

        return (
            desired_roll,
            desired_pitch,
            desired_z_acceleration
        )