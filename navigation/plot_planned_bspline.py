import numpy as np
import matplotlib.pyplot as plt

from navigation.obstacle_planner import (
    BoxObstacle,
    inflate_obstacle
)

from navigation.waypoint_planner import (
    find_shortest_collision_free_path
)

from navigation.bspline import (
    generate_bspline
)


# ==========================================================
# PATH LENGTH
# ==========================================================

def calculate_path_length(path):
    total_length = 0.0

    for i in range(len(path) - 1):
        segment = path[i + 1] - path[i]
        total_length += np.linalg.norm(segment)

    return total_length


# ==========================================================
# DRAW BOX
# ==========================================================

def draw_box(ax, obstacle, alpha=0.25):
    minimum = obstacle.minimum
    maximum = obstacle.maximum

    x = [
        minimum[0],
        maximum[0]
    ]

    y = [
        minimum[1],
        maximum[1]
    ]

    z = [
        minimum[2],
        maximum[2]
    ]

    # Bottom rectangle
    xx, yy = np.meshgrid(x, y)

    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, z[0]),
        alpha=alpha
    )

    # Top rectangle
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, z[1]),
        alpha=alpha
    )

    # Four vertical edges
    for xi in x:
        for yi in y:

            ax.plot(
                [xi, xi],
                [yi, yi],
                z
            )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print()
    print("==================================================")
    print("B-SPLINE TRAJECTORY VISUALIZATION")
    print("==================================================")


    # ======================================================
    # START AND GOAL
    # ======================================================

    start = np.array(
        [
            0.0,
            0.0,
            1.0
        ]
    )

    goal = np.array(
        [
            4.0,
            0.0,
            1.0
        ]
    )


    # ======================================================
    # OBSTACLE
    # ======================================================

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


    # ======================================================
    # SAFETY DISTANCE
    # ======================================================

    safety_distance = 0.25


    # ======================================================
    # SELECTED PLANNING CLEARANCE
    # ======================================================
    #
    # This value was obtained by the previous search:
    #
    # 0.05 -> unsafe
    # 0.10 -> unsafe
    # 0.15 -> unsafe
    # 0.20 -> unsafe
    # 0.25 -> unsafe
    # 0.30 -> unsafe
    # 0.35 -> safe
    #
    # Therefore 0.35 m is the first successful
    # tested clearance for this scenario.
    #
    # ======================================================

    planning_clearance = 0.35


    # ======================================================
    # B-SPLINE PARAMETERS
    # ======================================================

    degree = 3

    number_of_samples = 401


    # ======================================================
    # PLAN WAYPOINTS
    # ======================================================

    waypoints, waypoint_length = (
        find_shortest_collision_free_path(
            start,
            goal,
            [obstacle],
            safety_distance,
            planning_clearance
        )
    )


    # ======================================================
    # GENERATE B-SPLINE
    # ======================================================

    parameters, trajectory, knots = (
        generate_bspline(
            waypoints,
            number_of_samples=number_of_samples,
            degree=degree
        )
    )


    # ======================================================
    # PRINT INFORMATION
    # ======================================================

    bspline_length = calculate_path_length(
        trajectory
    )


    inflated_obstacle = inflate_obstacle(
        obstacle,
        safety_distance
    )


    print()
    print("WAYPOINTS")
    print("----------------------------------------------")

    for i, waypoint in enumerate(waypoints):

        print(
            f"P{i} = "
            f"[{waypoint[0]:.3f}, "
            f"{waypoint[1]:.3f}, "
            f"{waypoint[2]:.3f}]"
        )


    print()
    print("B-SPLINE")
    print("----------------------------------------------")

    print(
        "Degree =",
        degree
    )

    print(
        "Samples =",
        number_of_samples
    )

    print(
        "Path length =",
        f"{bspline_length:.6f} m"
    )


    print()
    print("INFLATED OBSTACLE")
    print("----------------------------------------------")

    print(
        "Minimum =",
        inflated_obstacle.minimum
    )

    print(
        "Maximum =",
        inflated_obstacle.maximum
    )


    # ======================================================
    # CREATE FIGURE
    # ======================================================

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # ======================================================
    # ORIGINAL OBSTACLE
    # ======================================================

    draw_box(
        ax,
        obstacle,
        alpha=0.35
    )


    # ======================================================
    # INFLATED OBSTACLE
    # ======================================================

    draw_box(
        ax,
        inflated_obstacle,
        alpha=0.12
    )


    # ======================================================
    # WAYPOINT PATH
    # ======================================================

    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        marker="o",
        linewidth=2,
        label="Waypoint path"
    )


    # ======================================================
    # B-SPLINE
    # ======================================================

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        linewidth=3,
        label="Cubic B-spline"
    )


    # ======================================================
    # START
    # ======================================================

    ax.scatter(
        start[0],
        start[1],
        start[2],
        s=80,
        marker="o",
        label="Start"
    )


    # ======================================================
    # GOAL
    # ======================================================

    ax.scatter(
        goal[0],
        goal[1],
        goal[2],
        s=80,
        marker="X",
        label="Goal"
    )


    # ======================================================
    # LABEL WAYPOINTS
    # ======================================================

    for i, point in enumerate(waypoints):

        ax.text(
            point[0],
            point[1],
            point[2] + 0.08,
            f"P{i}"
        )


    # ======================================================
    # AXES
    # ======================================================

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "Cubic B-Spline Obstacle-Avoidance Trajectory"
    )


    # ======================================================
    # LEGEND
    # ======================================================

    ax.legend()


    # ======================================================
    # VIEW
    # ======================================================

    ax.view_init(
        elev=25,
        azim=-60
    )


    plt.tight_layout()

    plt.show()