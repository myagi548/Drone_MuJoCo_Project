import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD SAVED FREE-RUNNING RESULTS
# ==========================================================

actual = np.load(
    "results/dmdc_free_run_actual.npy"
)

predicted = np.load(
    "results/dmdc_free_run_predicted.npy"
)


# ==========================================================
# 2. TIME VECTOR
# ==========================================================

dt = 0.002

number_of_samples = actual.shape[0]

time = np.arange(
    number_of_samples
) * dt


# ==========================================================
# 3. STATE NAMES
# ==========================================================

state_names = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r"
]


# ==========================================================
# 4. POSITION PLOTS
# ==========================================================

position_indices = [0, 1, 2]

for index in position_indices:

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        actual[:, index],
        label="MuJoCo actual"
    )

    plt.plot(
        time,
        predicted[:, index],
        "--",
        label="DMDc predicted"
    )

    plt.xlabel("Time (s)")
    plt.ylabel(
        state_names[index]
    )

    plt.title(
        f"{state_names[index].upper()} Position: "
        "MuJoCo vs Free-Running DMDc"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ==========================================================
# 5. ATTITUDE PLOTS
# ==========================================================

attitude_indices = [6, 7, 8]

for index in attitude_indices:

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        np.degrees(actual[:, index]),
        label="MuJoCo actual"
    )

    plt.plot(
        time,
        np.degrees(predicted[:, index]),
        "--",
        label="DMDc predicted"
    )

    plt.xlabel("Time (s)")

    plt.ylabel(
        f"{state_names[index]} (degrees)"
    )

    plt.title(
        f"{state_names[index].upper()} Attitude: "
        "MuJoCo vs Free-Running DMDc"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ==========================================================
# 6. VELOCITY PLOTS
# ==========================================================

velocity_indices = [3, 4, 5]

for index in velocity_indices:

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        actual[:, index],
        label="MuJoCo actual"
    )

    plt.plot(
        time,
        predicted[:, index],
        "--",
        label="DMDc predicted"
    )

    plt.xlabel("Time (s)")

    plt.ylabel(
        f"{state_names[index]} (m/s)"
    )

    plt.title(
        f"{state_names[index].upper()}: "
        "MuJoCo vs Free-Running DMDc"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ==========================================================
# 7. ANGULAR RATE PLOTS
# ==========================================================

rate_indices = [9, 10, 11]

for index in rate_indices:

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        actual[:, index],
        label="MuJoCo actual"
    )

    plt.plot(
        time,
        predicted[:, index],
        "--",
        label="DMDc predicted"
    )

    plt.xlabel("Time (s)")

    plt.ylabel(
        f"{state_names[index]} (rad/s)"
    )

    plt.title(
        f"{state_names[index].upper()}: "
        "MuJoCo vs Free-Running DMDc"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


print()
print("==================================================")
print("DMDc FREE-RUNNING PLOTS")
print("==================================================")
print()
print("Actual data shape    :", actual.shape)
print("Predicted data shape :", predicted.shape)
print(
    f"Time duration        : {time[-1]:.3f} s"
)
print()
print("Plots generated successfully.")
print()