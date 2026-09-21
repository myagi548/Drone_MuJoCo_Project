
import numpy as np

from navigation.obstacle_planner import (
    BoxObstacle,
    path_is_collision_free,
    find_collision,
)

# --------------------------------------------------
# 1. OBSTACLES FROM drone_multi_obstacle.xml
# MuJoCo size values are half-sizes.
# --------------------------------------------------

OBSTACLES = [
    BoxObstacle(
        center=[2.0, 0.0, 1.0],
        size=[0.8, 0.8, 2.0]
    ),
    BoxObstacle(
        center=[4.0, 1.5, 1.0],
        size=[0.8, 0.8, 2.0]
    ),
    BoxObstacle(
        center=[6.0, -1.5, 1.0],
        size=[0.8, 0.8, 2.0]
    ),
    BoxObstacle(
        center=[8.0, 0.0, 1.0],
        size=[0.8, 0.8, 2.0]
    ),
]

# --------------------------------------------------
# 2. EXACT WAYPOINTS FROM YOUR NO-WIND SCRIPT
# --------------------------------------------------

WAYPOINTS = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [3.0, 1.0, 1.0],
    [3.0, 0.0, 1.0],
    [3.0, 2.5, 1.0],
    [5.0, 2.5, 1.0],
    [5.0, 0.0, 1.0],
    [5.0, -2.5, 1.0],
    [7.0, -2.5, 1.0],
    [7.0, 0.0, 1.0],
    [7.0, 1.0, 1.0],
    [9.0, 1.0, 1.0],
    [10.0, 0.0, 1.0],
])

# --------------------------------------------------
# 3. DRONE CLEARANCE
# Arm length = 0.45 m
# Capsule radius = 0.035 m
# --------------------------------------------------

DRONE_CLEARANCE = 0.485

# --------------------------------------------------
# 4. RUN COLLISION CHECKS
# --------------------------------------------------

print()
print("=" * 55)
print("MULTI-OBSTACLE ROUTE VALIDATION")
print("=" * 55)

print("Number of waypoints:", len(WAYPOINTS))
print("Number of obstacles:", len(OBSTACLES))
print("Clearance allowance:", DRONE_CLEARANCE, "m")

# Check the waypoint centerline.
collision = find_collision(
    WAYPOINTS,
    OBSTACLES
)

# Check segments with inflated obstacles.
clear = path_is_collision_free(
    WAYPOINTS,
    OBSTACLES,
    safety_distance=DRONE_CLEARANCE
)

print()
print(
    "Centerline collision check:",
    "PASS" if collision is None else "FAIL"
)

print(
    "Inflated-box clearance check:",
    "PASS" if clear else "FAIL"
)

# Report the first collision, if found.
if collision is not None:

    segment_index, obstacle_index = collision

    print()
    print("First potential collision:")
    print("Segment:", segment_index + 1)
    print("From:", WAYPOINTS[segment_index])
    print("To:", WAYPOINTS[segment_index + 1])
    print("Obstacle:", obstacle_index + 1)

print()
print("=" * 55)

if clear:
    print("RESULT: WAYPOINT SEGMENTS CLEAR THE INFLATED BOXES")
else:
    print("RESULT: POTENTIAL COLLISION OR CLEARANCE ISSUE FOUND")

print("=" * 55)