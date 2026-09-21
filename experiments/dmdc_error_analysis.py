import numpy as np


# ==========================================================
# LOAD RESULTS
# ==========================================================

actual = np.load(
    "results/dmdc_free_run_actual.npy"
)

predicted = np.load(
    "results/dmdc_free_run_predicted.npy"
)


# ==========================================================
# TIME
# ==========================================================

dt = 0.002

time = np.arange(
    actual.shape[0]
) * dt


# ==========================================================
# ERROR
# ==========================================================

error = actual - predicted


# ==========================================================
# STATE NAMES
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
# OVERALL RMSE AT DIFFERENT HORIZONS
# ==========================================================

print()
print("==================================================")
print("DMDc PREDICTION ERROR GROWTH")
print("==================================================")


check_times = [
    0.5,
    1.0,
    1.5,
    1.998
]


print()
print("OVERALL RMSE VS PREDICTION TIME")
print("----------------------------------------------")


for check_time in check_times:

    index = int(
        check_time / dt
    )

    index = min(
        index,
        len(error) - 1
    )

    current_error = error[
        :index + 1
    ]

    current_rmse = np.sqrt(
        np.mean(
            current_error ** 2
        )
    )

    print(
        f"{time[index]:.3f} s : "
        f"{current_rmse:.12e}"
    )


# ==========================================================
# FINAL STATE RMSE
# ==========================================================

final_rmse = np.sqrt(
    np.mean(
        error ** 2,
        axis=0
    )
)


print()
print("FINAL STATE RMSE")
print("----------------------------------------------")


for name, value in zip(
    state_names,
    final_rmse
):

    print(
        f"{name:>5s} : "
        f"{value:.12e}"
    )


# ==========================================================
# MAXIMUM ABSOLUTE ERROR
# ==========================================================

maximum_error = np.max(
    np.abs(error),
    axis=0
)


print()
print("MAXIMUM ABSOLUTE ERROR")
print("----------------------------------------------")


for name, value in zip(
    state_names,
    maximum_error
):

    print(
        f"{name:>5s} : "
        f"{value:.12e}"
    )


# ==========================================================
# FINAL ERROR
# ==========================================================

print()
print("FINAL ERROR")
print("----------------------------------------------")

print(
    error[-1]
)


# ==========================================================
# CONVERSION OF ATTITUDE ERRORS TO DEGREES
# ==========================================================

print()
print("ATTITUDE ERROR IN DEGREES")
print("----------------------------------------------")


for index, name in zip(
    [6, 7, 8],
    ["roll", "pitch", "yaw"]
):

    final_error_deg = np.degrees(
        abs(error[-1, index])
    )

    max_error_deg = np.degrees(
        maximum_error[index]
    )

    print(
        f"{name:>5s} : "
        f"final = {final_error_deg:.9e} deg, "
        f"maximum = {max_error_deg:.9e} deg"
    )


print()
print("==================================================")
print("ERROR ANALYSIS COMPLETE")
print("==================================================")