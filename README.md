# Autonomous Self-Stabilizing Drone with Obstacle Avoidance

<p align="center">
  <img src="assets/amrita_logo.png" alt="Amrita Vishwa Vidyapeetham Logo" width="180">
</p>

<p align="center">
  <strong>Amrita Vishwa Vidyapeetham</strong><br>
  Coimbatore, Tamil Nadu, India<br>
  Department of Artificial Intelligence
</p>

---

## 1. Project Overview

This project develops an autonomous self-stabilizing drone simulation using MuJoCo. The drone is designed to stabilize its attitude, follow a smooth B-spline trajectory, detect and avoid obstacles, predict unsafe flight conditions, and recover from attitude disturbances.

The project combines PID control, trajectory planning, obstacle avoidance, state-space analysis, and machine-learning-based safety prediction.

## 2. Project Objectives

- Implement drone simulation using MuJoCo.
- Stabilize roll and pitch using PID control.
- Generate smooth trajectories using cubic B-splines.
- Detect obstacles and generate collision-free paths.
- Analyze system observability and controllability.
- Predict unsafe attitude conditions using Logistic Regression.
- Recover from unsafe attitude conditions.
- Evaluate flight under wind disturbances.
- Extend navigation to multiple obstacles and targets.

## 3. System Workflow

```text
Drone Initialization
        |
        v
Stable Hover
        |
        v
B-Spline Trajectory Generation
        |
        v
Forward Movement
        |
        v
Obstacle Detection
        |
        v
Collision-Free Path Planning
        |
        v
Obstacle Avoidance
        |
        v
Safety Prediction
        |
        +---- SAFE ------> Continue Trajectory
        |
        +---- UNSAFE ----> Recovery Controller
                                  |
                                  v
                         Attitude Stabilization
                                  |
                                  v
                         Resume Trajectory
                                  |
                                  v
                            Target Reached
```

This diagram represents the intended workflow. Each stage must be validated through its corresponding simulation experiment.

## 4. Technologies Used

| Component | Technology |
|---|---|
| Programming | Python |
| Physics simulation | MuJoCo |
| Numerical computation | NumPy |
| Trajectory planning | Cubic B-splines |
| Flight control | PID |
| Safety prediction | Logistic Regression |
| Visualization | MuJoCo viewer and plotting tools |

## 5. Repository Structure

```text
Drone_MuJoCo_Project/
|
+-- analysis/
+-- control/
+-- experiments/
+-- navigation/
|   +-- bspline.py
|   +-- b_spline_trajectory.py
|   +-- obstacle_planner.py
|   +-- planned_bspline.py
|   +-- plot_planned_bspline.py
|   +-- waypoint_planner.py
+-- prediction/
+-- results/
+-- sensors/
+-- simulation/
|   +-- drone_model.xml
|   +-- mixer.py
|   +-- mujoco_force_assignment_test.py
|   +-- pid_mixer_mujoco_test.py
|   +-- pid_mixer_mujoco_viewer.py
|   +-- pid_roll_test.py
|   +-- position_bspline_mujoco_test.py
+-- assets/
|   +-- amrita_logo.png
+-- README.md
+-- .gitignore
```

## 6. Mathematical Model

### 6.1 Rotational Dynamics

The simplified rotational dynamics about one axis are:

```text
I * theta_ddot(t) = tau(t)
```

Taking the Laplace transform with zero initial conditions:

```text
I * s^2 * Theta(s) = Tau(s)
```

The transfer function is:

```text
G(s) = Theta(s) / Tau(s) = 1 / (I * s^2)
```

Here, `I` is the moment of inertia, `theta` is angular position, and `tau` is applied torque.

This is a simplified single-axis model. A full drone model includes coupled rotational and translational dynamics.

### 6.2 PID Controller

The attitude error is:

```text
e(t) = desired_angle(t) - actual_angle(t)
```

The PID control law is:

```text
u(t) = Kp * e(t)
     + Ki * integral(e(lambda), lambda=0..t)
     + Kd * de(t)/dt
```

The proportional, integral, and derivative terms are:

```text
u_P(t) = Kp * e(t)
u_I(t) = Ki * integral(e(lambda), lambda=0..t)
u_D(t) = Kd * de(t)/dt
```

The total controller output is:

```text
u(t) = u_P(t) + u_I(t) + u_D(t)
```

### 6.3 PID Gain Derivation

For the plant:

```text
I * theta_ddot(t) = tau(t)
G(s) = 1 / (I * s^2)
```

Using a PID controller:

```text
C(s) = Kp + Ki/s + Kd*s
```

The closed-loop characteristic polynomial is:

```text
I*s^3 + Kd*s^2 + Kp*s + Ki = 0
```

Choose the desired characteristic polynomial:

```text
(s^2 + 2*zeta*wn*s + wn^2) * (s + p3)
```

Expanding:

```text
s^3
+ (2*zeta*wn + p3)*s^2
+ (wn^2 + 2*zeta*wn*p3)*s
+ wn^2*p3
```

Matching coefficients gives:

```text
Kd = I * (2*zeta*wn + p3)
Kp = I * (wn^2 + 2*zeta*wn*p3)
Ki = I * wn^2 * p3
```

Here, `zeta` is the damping ratio, `wn` is the natural frequency, and `p3` is the additional real pole.

These formulas apply to the simplified plant and stated controller structure. The gains must be validated against the actual MuJoCo dynamics and actuator limits.

## 7. B-Spline Trajectory Planning

### 7.1 General B-Spline Curve

A B-spline curve of degree `p` is:

```text
P(u) = sum(N_i,p(u) * P_i), for i = 0..n
```

Here, `P_i` are control points, `N_i,p(u)` are basis functions, and `u` is the curve parameter.

### 7.2 Degree-Zero Basis Function

```text
N_i,0(u) = 1, when t_i <= u < t_(i+1)
N_i,0(u) = 0, otherwise
```

Here, `t_i` and `t_(i+1)` are consecutive knot values.

### 7.3 Cox-de Boor Recursion

For `p > 0`:

```text
N_i,p(u) =
    ((u - t_i) / (t_(i+p) - t_i)) * N_i,p-1(u)
  + ((t_(i+p+1) - u) / (t_(i+p+1) - t_(i+1))) * N_(i+1),p-1(u)
```

A term with a zero denominator is taken as zero.

### 7.4 Cubic B-Spline

For a cubic B-spline:

```text
p = 3
P(u) = sum(N_i,3(u) * P_i), for i = 0..n
```

The first and second derivatives describe parameter-space velocity and acceleration:

```text
V(u) = dP(u)/du
A(u) = d^2P(u)/du^2
```

For time-parameterized motion `u = u(t)`:

```text
dP/dt = (dP/du) * (du/dt)
```

The trajectory must be checked for obstacle clearance after spline construction.

## 8. Path Planning and Obstacle Avoidance

### 8.1 Straight-Line Reference Path

A straight-line path from start to goal is:

```text
P(lambda) = P_s + lambda * (P_g - P_s)
0 <= lambda <= 1
```

For `N` intervals:

```text
lambda_k = k/N, for k = 0, 1, ..., N
P_k = P_s + (k/N) * (P_g - P_s)
```

### 8.2 Axis-Aligned Box Obstacle

For obstacle center `c` and size vector `d`:

```text
b_min = c - d/2
b_max = c + d/2
```

Inflating the obstacle by safety distance `d_s`:

```text
b_min_inflated = b_min - d_s
b_max_inflated = b_max + d_s
```

A candidate path is accepted only if it maintains the required clearance from the inflated obstacle. The complete spline trajectory should be checked, not only its waypoints.

## 9. State-Space Analysis

### 9.1 State-Space Model

A linear system is represented by:

```text
x_dot = A*x + B*u
y = C*x + D*u
```

Here, `x` is the state vector, `u` is the input, and `y` is the output.

### 9.2 Observability

The observability matrix is:

```text
O = [ C
      C*A
      C*A^2
      ...
      C*A^(n-1) ]
```

The system is observable if:

```text
rank(O) = n
```

Here, `n` is the number of states.

### 9.3 Controllability

The controllability matrix is:

```text
Ctr = [ B  A*B  A^2*B  ...  A^(n-1)*B ]
```

The system is controllable if:

```text
rank(Ctr) = n
```

The matrices `A`, `B`, and `C` must correspond to the actual state and measurement definitions used in the simulation.

## 10. Logistic Regression Safety Prediction

The Logistic Regression model estimates the probability of a designated safety class:

```text
P(y=1 | x) = 1 / (1 + exp(-z))
z = w^T*x + b
```

For a feature vector with `m` features:

```text
z = sum(w_j*x_j, j=1..m) + b
```

The predicted class is defined by the chosen threshold `T`:

```text
If P(y=1 | x) >= T:
    predicted class = 1
Otherwise:
    predicted class = 0
```

Document explicitly whether class `1` means SAFE or UNSAFE. Possible input features include roll, pitch, angular velocity, and trajectory tracking error. The actual feature set, labels, threshold, and model performance must be documented after training and testing.

## 11. Recovery Control

The recovery controller is intended to bring the drone back toward its desired attitude when an unsafe condition is detected.

For roll and pitch:

```text
e_roll(t) = desired_roll(t) - roll(t)
e_pitch(t) = desired_pitch(t) - pitch(t)
```

The corresponding PID outputs are:

```text
u_roll(t) =
    Kp_roll*e_roll(t)
  + Ki_roll*integral(e_roll(lambda), lambda=0..t)
  + Kd_roll*de_roll(t)/dt

u_pitch(t) =
    Kp_pitch*e_pitch(t)
  + Ki_pitch*integral(e_pitch(lambda), lambda=0..t)
  + Kd_pitch*de_pitch(t)/dt
```

The controller should resume trajectory tracking only after the defined recovery conditions are satisfied.

## 12. Wind Disturbance Scenario

The intended wind-response sequence is:

```text
Wind Disturbance
       |
       v
Measure State Deviation
       |
       v
Stabilize Attitude
       |
       v
Check Stability Conditions
       |
       v
Resume B-Spline Trajectory
```

A simplified rotational model with an external disturbance torque is:

```text
I * theta_ddot(t) = tau_c(t) + tau_w(t)
```

Here, `tau_c(t)` is controller torque and `tau_w(t)` is disturbance torque caused by wind.

The controller aims to reduce the attitude error:

```text
e(t) = desired_angle(t) - actual_angle(t)
```

Wind response should be evaluated using recorded attitude, position error, and recovery time.

## 13. Installation

Activate the Conda environment:

```powershell
conda activate drone_mujoco
```

Navigate to the project folder:

```powershell
cd C:\Users\patim\Documents\Drone_MuJoCo_Project
```

If a `requirements.txt` file exists, install its dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 14. Running the Simulation

Run commands from the project root directory.

Example attitude test:

```powershell
python simulation\pid_roll_test.py
```

Example navigation module:

```powershell
python -m navigation.waypoint_planner
```

Use the appropriate script for each experiment. If a command fails, resolve that error before proceeding.

## 15. Experiments

| Experiment | Objective |
|---|---|
| Attitude stabilization | Evaluate PID roll and pitch control |
| Rotor force assignment | Verify force application |
| Position control | Evaluate position tracking |
| B-spline trajectory | Generate and track a smooth path |
| Single obstacle | Test basic obstacle avoidance |
| Multiple obstacles | Test path planning in a more complex environment |
| Multiple targets | Test sequential target navigation |
| Wind disturbance | Evaluate stabilization and trajectory resumption |
| Safety prediction | Evaluate unsafe-state classification |
| Recovery controller | Evaluate attitude recovery |

## 16. Results

Simulation results, plots, and performance measurements will be added after the corresponding experiments have been executed and verified.

No unverified performance values are reported here.

## 17. Future Work

- Complete and validate the baseline simulation.
- Extend obstacle avoidance to multiple obstacles.
- Extend navigation to multiple targets.
- Implement wind stabilization and trajectory resumption.
- Integrate and evaluate the safety prediction model.
- Validate recovery behavior under disturbances.
- Record reproducible results and prepare experiment documentation.

---

<p align="center">
  <strong>Autonomous Self-Stabilizing Drone with Obstacle Avoidance</strong><br>
  Amrita Vishwa Vidyapeetham - Coimbatore
</p>
