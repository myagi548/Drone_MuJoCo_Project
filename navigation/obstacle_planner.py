import numpy as np


# ==========================================================
# OBSTACLE REPRESENTATION
# ==========================================================

class BoxObstacle:
    """
    Axis-aligned rectangular obstacle.

    center = [x, y, z]
    size   = [length_x, length_y, height_z]
    """

    def __init__(self, center, size):

        self.center = np.asarray(
            center,
            dtype=float
        )

        self.size = np.asarray(
            size,
            dtype=float
        )

        if self.center.shape != (3,):

            raise ValueError(
                "Obstacle center must contain "
                "exactly 3 values."
            )

        if self.size.shape != (3,):

            raise ValueError(
                "Obstacle size must contain "
                "exactly 3 values."
            )

        if np.any(self.size <= 0):

            raise ValueError(
                "Obstacle dimensions must "
                "be positive."
            )


    # ------------------------------------------------------
    # Minimum and maximum coordinates
    # ------------------------------------------------------

    @property
    def minimum(self):

        return (
            self.center
            - self.size / 2.0
        )


    @property
    def maximum(self):

        return (
            self.center
            + self.size / 2.0
        )


# ==========================================================
# POINT INSIDE OBSTACLE
# ==========================================================

def point_inside_obstacle(
    point,
    obstacle
):
    """
    Check whether a point lies inside
    the obstacle.
    """

    point = np.asarray(
        point,
        dtype=float
    )

    return np.all(
        point >= obstacle.minimum
    ) and np.all(
        point <= obstacle.maximum
    )


# ==========================================================
# INFLATED OBSTACLE
# ==========================================================

def inflate_obstacle(
    obstacle,
    safety_distance
):
    """
    Increase obstacle dimensions by
    2 * safety_distance.

    This accounts for the required clearance
    between the drone and the physical obstacle.
    """

    if safety_distance < 0:

        raise ValueError(
            "Safety distance cannot be negative."
        )


    inflated_size = (
        obstacle.size
        + 2.0 * safety_distance
    )


    return BoxObstacle(
        obstacle.center,
        inflated_size
    )


# ==========================================================
# LINE SEGMENT COLLISION CHECK
# ==========================================================

def segment_intersects_obstacle(
    start,
    end,
    obstacle
):
    """
    Check whether the straight line segment
    from start to end intersects the obstacle.

    Uses the slab intersection method.
    """

    start = np.asarray(
        start,
        dtype=float
    )

    end = np.asarray(
        end,
        dtype=float
    )


    direction = end - start


    t_min = 0.0
    t_max = 1.0


    for axis in range(3):

        # --------------------------------------------------
        # Segment is parallel to this coordinate axis.
        # --------------------------------------------------

        if abs(direction[axis]) < 1e-12:

            if (
                start[axis]
                < obstacle.minimum[axis]
                or
                start[axis]
                > obstacle.maximum[axis]
            ):

                return False

            continue


        # --------------------------------------------------
        # Calculate intersection parameters.
        # --------------------------------------------------

        t1 = (
            obstacle.minimum[axis]
            - start[axis]
        ) / direction[axis]


        t2 = (
            obstacle.maximum[axis]
            - start[axis]
        ) / direction[axis]


        if t1 > t2:

            t1, t2 = t2, t1


        t_min = max(
            t_min,
            t1
        )

        t_max = min(
            t_max,
            t2
        )


        if t_min > t_max:

            return False


    return True


# ==========================================================
# PATH COLLISION CHECK
# ==========================================================

def path_is_collision_free(
    path,
    obstacles,
    safety_distance=0.0
):
    """
    Check every consecutive path segment
    against every obstacle.

    Returns
    -------
    bool
        True  -> path is collision free
        False -> collision detected
    """

    path = np.asarray(
        path,
        dtype=float
    )


    if path.ndim != 2:

        raise ValueError(
            "Path must have shape (N, 3)."
        )


    if path.shape[1] != 3:

        raise ValueError(
            "Path must contain x, y, z."
        )


    # ------------------------------------------------------
    # Inflate every obstacle.
    # ------------------------------------------------------

    checked_obstacles = [
        inflate_obstacle(
            obstacle,
            safety_distance
        )
        for obstacle in obstacles
    ]


    # ------------------------------------------------------
    # Check every path segment.
    # ------------------------------------------------------

    for i in range(
        len(path) - 1
    ):

        start = path[i]

        end = path[i + 1]


        for obstacle in checked_obstacles:

            if segment_intersects_obstacle(
                start,
                end,
                obstacle
            ):

                return False


    return True


# ==========================================================
# FIND FIRST COLLIDING SEGMENT
# ==========================================================

def find_collision(
    path,
    obstacles,
    safety_distance=0.0
):
    """
    Return information about the first
    collision.

    Returns None when no collision exists.
    """

    path = np.asarray(
        path,
        dtype=float
    )


    checked_obstacles = [
        inflate_obstacle(
            obstacle,
            safety_distance
        )
        for obstacle in obstacles
    ]


    for segment_index in range(
        len(path) - 1
    ):

        start = path[segment_index]

        end = path[segment_index + 1]


        for obstacle_index, obstacle in enumerate(
            checked_obstacles
        ):

            if segment_intersects_obstacle(
                start,
                end,
                obstacle
            ):

                return (
                    segment_index,
                    obstacle_index
                )


    return None


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("==================================================")
    print("OBSTACLE COLLISION CHECK TEST")
    print("==================================================")


    # ------------------------------------------------------
    # Test obstacle
    #
    # These values are test values only.
    # They are not yet the final MuJoCo environment.
    # ------------------------------------------------------

    obstacle = BoxObstacle(
        center=[
            2.0,
            0.0,
            1.0
        ],
        size=[
            1.0,
            1.0,
            1.0
        ]
    )


    print()
    print("OBSTACLE")
    print("----------------------------------------------")

    print(
        "Center =",
        obstacle.center
    )

    print(
        "Size   =",
        obstacle.size
    )

    print(
        "Minimum =",
        obstacle.minimum
    )

    print(
        "Maximum =",
        obstacle.maximum
    )


    # ------------------------------------------------------
    # Test points
    # ------------------------------------------------------

    inside_point = np.array(
        [
            2.0,
            0.0,
            1.0
        ]
    )


    outside_point = np.array(
        [
            0.0,
            0.0,
            1.0
        ]
    )


    print()
    print("POINT TEST")
    print("----------------------------------------------")

    print(
        "Inside point:",
        inside_point
    )

    print(
        "Result:",
        point_inside_obstacle(
            inside_point,
            obstacle
        )
    )


    print()

    print(
        "Outside point:",
        outside_point
    )

    print(
        "Result:",
        point_inside_obstacle(
            outside_point,
            obstacle
        )
    )


    # ------------------------------------------------------
    # Straight path through obstacle
    # ------------------------------------------------------

    path_through = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 0.0, 1.0]
        ]
    )


    print()
    print("PATH 1: THROUGH OBSTACLE")
    print("----------------------------------------------")

    collision = find_collision(
        path_through,
        [obstacle]
    )

    print(
        "Collision information:",
        collision
    )

    print(
        "Collision free:",
        path_is_collision_free(
            path_through,
            [obstacle]
        )
    )


    # ------------------------------------------------------
    # Path above obstacle
    # ------------------------------------------------------

    path_above = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
            [4.0, 0.0, 1.0]
        ]
    )


    print()
    print("PATH 2: ABOVE OBSTACLE")
    print("----------------------------------------------")

    collision = find_collision(
        path_above,
        [obstacle]
    )

    print(
        "Collision information:",
        collision
    )

    print(
        "Collision free:",
        path_is_collision_free(
            path_above,
            [obstacle]
        )
    )


    # ------------------------------------------------------
    # Safety-distance test
    # ------------------------------------------------------

    safety_distance = 0.25


    inflated = inflate_obstacle(
        obstacle,
        safety_distance
    )


    print()
    print("INFLATED OBSTACLE")
    print("----------------------------------------------")

    print(
        "Safety distance =",
        safety_distance,
        "m"
    )

    print(
        "Original size =",
        obstacle.size
    )

    print(
        "Inflated size =",
        inflated.size
    )

    print(
        "Inflated minimum =",
        inflated.minimum
    )

    print(
        "Inflated maximum =",
        inflated.maximum
    )


    print()
    print("==================================================")
    print("OBSTACLE TEST COMPLETE")
    print("==================================================")