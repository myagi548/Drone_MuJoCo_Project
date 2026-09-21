import csv
import matplotlib.pyplot as plt


# ==========================================================
# 1. FILE LOCATION
# ==========================================================

input_file = "results/pid_roll_response.csv"


# ==========================================================
# 2. READ CSV DATA
# ==========================================================

times = []
roll_angles = []

with open(input_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        times.append(
            float(row["time"])
        )

        roll_angles.append(
            float(row["roll_deg"])
        )


# ==========================================================
# 3. MEASUREMENT VALUES
# ==========================================================

initial_roll = abs(
    roll_angles[0]
)

settling_band = (
    0.02 * initial_roll
)


# ==========================================================
# 4. FIND OVERSHOOT
# ==========================================================

negative_rolls = [
    angle
    for angle in roll_angles
    if angle < 0.0
]

if negative_rolls:

    overshoot_angle = abs(
        min(negative_rolls)
    )

    overshoot_percent = (
        overshoot_angle
        / initial_roll
        * 100.0
    )

else:

    overshoot_angle = 0.0
    overshoot_percent = 0.0


# ==========================================================
# 5. FIND SETTLING TIME
# ==========================================================

settling_time = None

for i in range(len(times)):

    remaining_values = [
        abs(angle)
        for angle in roll_angles[i:]
    ]

    if all(
        value <= settling_band
        for value in remaining_values
    ):

        settling_time = times[i]

        break


# ==========================================================
# 6. PRINT RESULTS
# ==========================================================

print()
print("==============================================")
print("PID RESPONSE PLOT")
print("==============================================")

print(
    f"Initial roll       = {initial_roll:.4f} deg"
)

print(
    f"Settling band      = +/-{settling_band:.4f} deg"
)

print(
    f"Overshoot angle    = {overshoot_angle:.4f} deg"
)

print(
    f"Overshoot          = {overshoot_percent:.4f} %"
)

if settling_time is not None:

    print(
        f"Settling time      = {settling_time:.4f} s"
    )

else:

    print(
        "Settling time      = Not reached"
    )

print("==============================================")


# ==========================================================
# 7. PLOT
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    times,
    roll_angles,
    label="Actual roll"
)

plt.axhline(
    0.0,
    linestyle="--",
    label="Desired roll = 0 deg"
)

plt.axhline(
    settling_band,
    linestyle=":",
    label=f"+{settling_band:.2f} deg settling band"
)

plt.axhline(
    -settling_band,
    linestyle=":",
    label=f"-{settling_band:.2f} deg settling band"
)


# Mark overshoot

if negative_rolls:

    overshoot_index = roll_angles.index(
        min(negative_rolls)
    )

    plt.scatter(
        times[overshoot_index],
        roll_angles[overshoot_index],
        label=(
            f"Overshoot = "
            f"{overshoot_percent:.2f}%"
        )
    )


# Mark settling time

if settling_time is not None:

    plt.axvline(
        settling_time,
        linestyle="--",
        label=(
            f"Settling time = "
            f"{settling_time:.2f} s"
        )
    )


plt.xlabel("Time (s)")

plt.ylabel("Roll angle (degrees)")

plt.title(
    "MuJoCo Drone PID Roll Response"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()