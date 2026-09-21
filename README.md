# Autonomous Self-Stabilizing Drone with Obstacle Avoidance

<div align="center">

<img src="assets/amrita_logo.png" alt="Amrita Vishwa Vidyapeetham Logo" width="180">

# AMRITA VISHWA VIDYAPEETHAM

**Coimbatore, Tamil Nadu**

**Department of Computer Science and Engineering (Artificial Intelligence)**

### Project Team — AB14

</div>

---

## Team Member Details

| S. No. | Name    | Roll Number      | Email                                                               |
| -----: | ------- | ---------------- | ------------------------------------------------------------------- |
|      1 | Devisri | CB.SC.U4AIE24163 | [devisri7142@gmail.com](mailto:devisri7142@gmail.com)               |
|      2 | Monisha | CB.SC.U4AIE24157 | [vemurimonishareddy@gmail.com](mailto:vemurimonishareddy@gmail.com) |
|      3 | Myagi   | CB.SC.U4AIE24143 | [patimamyagi@gmail.com](mailto:patimamyagi@gmail.com)               |

**Team Number:** AB14

---

## 1. Project Overview

This project focuses on the simulation of an autonomous self-stabilizing drone with obstacle avoidance using the MuJoCo physics simulator.

The system explores drone attitude stabilization, smooth trajectory generation, collision-free path planning, disturbance handling, and safety monitoring.

The intended drone behavior is to follow a planned trajectory, avoid obstacles, maintain a stable attitude, and recover from unsafe conditions.

## 2. Objectives

* Simulate drone dynamics using MuJoCo.
* Generate smooth trajectories using cubic B-splines.
* Plan collision-free paths around obstacles.
* Implement PID-based attitude and position control.
* Study drone stabilization under wind disturbances.
* Analyze controllability and observability using a mathematical model.
* Explore Logistic Regression for unsafe-state prediction.

## 3. System Architecture

```text
Start
  ↓
Initialize Drone in MuJoCo
  ↓
Generate Collision-Free Waypoints
  ↓
Generate Cubic B-Spline Trajectory
  ↓
Follow Trajectory Using PID Control
  ↓
Detect and Avoid Obstacles
  ↓
Monitor Attitude and Safety
  ↓
Recover if Attitude Becomes Unsafe
  ↓
Reach Target
```

## 4. PID Controller and Gain Calculation

A Proportional–Integral–Derivative (PID) controller calculates a control input using the difference between the desired and actual state.

### 4.1 Control Error

$$
e(t)=\theta_{\text{desired}}(t)-\theta_{\text{actual}}(t)
$$

where:

* $\theta_{\text{desired}}$ is the desired attitude angle.
* $\theta_{\text{actual}}$ is the measured attitude angle.
* $e(t)$ is the control error.

### 4.2 PID Control Equation

$$
u(t)=K_p e(t)+K_i\int_0^t e(\tau)\,d\tau
+K_d\frac{de(t)}{dt}
$$

where $K_p$, $K_i$, and $K_d$ are the proportional, integral, and derivative gains, respectively.

### 4.3 Simplified Drone Attitude Model

For a single rotational axis:

$$
I\ddot{\theta}=\tau
$$

where $I$ is the moment of inertia and $\tau$ is the applied torque.

The corresponding transfer function is:

$$
G(s)=\frac{\Theta(s)}{\Tau(s)}=\frac{1}{Is^2}
$$

### 4.4 Desired Closed-Loop Polynomial

For initial PID gain design, a desired characteristic polynomial can be selected as:

$$
(s^2+2\zeta\omega_n s+\omega_n^2)(s+p_3)
$$

Expanding:

$$
\begin{aligned}
& s^3+(2\zeta\omega_n+p_3)s^2\\
&+(\omega_n^2+2\zeta\omega_n p_3)s\\
&+\omega_n^2p_3
\end{aligned}
$$

The PID closed-loop characteristic polynomial for the simplified plant is:

$$
Is^3+K_ds^2+K_ps+K_i
$$

Matching the coefficients gives:

**Derivative gain**

$$
\boxed{K_d=I(2\zeta\omega_n+p_3)}
$$

**Proportional gain**

$$
\boxed{K_p=I(\omega_n^2+2\zeta\omega_n p_3)}
$$

**Integral gain**

$$
\boxed{K_i=I\omega_n^2p_3}
$$

### 4.5 Gain Design Parameters

The earlier gain-design example used:

| Parameter                     |                         Value |
| ----------------------------- | ----------------------------: |
| Damping ratio, $\zeta$        |                           0.7 |
| Natural frequency, $\omega_n$ |                     4.0 rad/s |
| Additional pole, $p_3$        |                     8.0 rad/s |
| Moment of inertia, $I$        | Obtained from the drone model |

The gains are calculated by substituting the moment of inertia from the MuJoCo model and the selected design parameters into the equations above.

These equations provide initial gains for the simplified model. The resulting controller must be evaluated in simulation.

## 5. Cubic B-Spline Trajectory Planning

B-splines generate smooth trajectories from a sequence of control points.

### 5.1 Straight-Line Initialization

A straight-line path between the start and goal positions is:

$$
P(\lambda)=P_s+\lambda(P_g-P_s)
$$

where $P_s$ is the start position, $P_g$ is the goal position, and $\lambda\in[0,1]$.

Sampling the path:

$$
\lambda_k=\frac{k}{N},
\qquad k=0,1,\ldots,N
$$

The sampled path can be checked for collisions before trajectory generation.

### 5.2 B-Spline Curve

A degree-$p$ B-spline curve is:

$$
C(u)=\sum_{i=0}^{n}N_{i,p}(u)P_i
$$

For a cubic B-spline, $p=3$.

Here:

* $P_i$ are the control points.
* $N_{i,p}(u)$ are the B-spline basis functions.
* $u$ is the spline parameter.
* $C(u)$ is the generated trajectory position.

### 5.3 Cox–de Boor Basis Recursion

The basis functions are defined recursively:

$$
\begin{aligned}
N_{i,p}(u)={}&
\frac{u-t_i}{t_{i+p}-t_i}N_{i,p-1}(u)\\
&+\frac{t_{i+p+1}-u}{t_{i+p+1}-t_{i+1}}
N_{i+1,p-1}(u)
\end{aligned}
$$

Terms with zero denominators are treated as zero.

The degree-zero basis function is:

$$
N_{i,0}(u)=
\begin{cases}
1,&t_i\leq u<t_{i+1}\\
0,&\text{otherwise}
\end{cases}
$$

where $t_i$ are the knot values.

### 5.4 Obstacle Inflation

For an axis-aligned box obstacle, the obstacle boundaries can be expanded by a clearance distance $d$:

$$
b_{\min}^{\text{inflated}}=b_{\min}-d
$$

$$
b_{\max}^{\text{inflated}}=b_{\max}+d
$$

The planner uses the expanded obstacle to account for clearance. The generated spline must also be collision-checked because the curve may deviate from the waypoint segments.

## 6. DMDC — Model-Based Control

The project investigates model-based control using a simplified rotational model.

$$
I\ddot{\theta}=\tau
$$

Define the state vector:

$$
x=
\begin{bmatrix}
\theta\\
\dot{\theta}
\end{bmatrix},
\qquad u=\tau
$$

The state-space model is:

$$
\dot{x}=
\begin{bmatrix}
0&1\\
0&0
\end{bmatrix}x+
\begin{bmatrix}
0\\
1/I
\end{bmatrix}u
$$

This model provides a mathematical basis for analyzing attitude dynamics and control inputs.

**Note:** The exact expansion and algorithm for DMDC should match the terminology and implementation used in the project.

## 7. DMDO — Observability Analysis

Observability describes whether a system's internal states can be inferred from its measured outputs.

For a linear system:

$$
\dot{x}=Ax+Bu,\qquad y=Cx
$$

the observability matrix is:

$$
\mathcal{O}=
\begin{bmatrix}
C\\
CA\\
CA^2\\
\vdots\\
CA^{n-1}
\end{bmatrix}
$$

The system is observable if:

$$
\boxed{\operatorname{rank}(\mathcal{O})=n}
$$

where $n$ is the number of states.

This condition applies to the selected linear model. It does not automatically establish observability for every state of the full nonlinear drone.

**Note:** Confirm the exact expansion and method for DMDO against the project's reference or implementation.

## 8. Wind Disturbance

Wind can be modeled as an external force acting on the drone.

For a sinusoidal disturbance:

$$
F_{\text{wind}}(t)=A\sin(\omega t+\phi)
$$

where:

* $A$ is the disturbance amplitude.
* $\omega$ is the disturbance frequency.
* $\phi$ is the phase.

The controller responds to disturbances by correcting attitude and position errors. In the planned wind scenario, the drone stabilizes before resuming trajectory following.

## 9. Machine-Learning Safety Prediction

The project explores Logistic Regression for classifying unsafe drone states.

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The predicted probability is:

$$
P(y=1\mid x)=\sigma(w^Tx+b)
$$

where $x$ represents input features, $w$ is the learned coefficient vector, and $b$ is the bias.

If the unsafe class is labeled $y=1$, the predicted probability can be compared against a classification threshold to determine the predicted state.

The dataset, selected features, threshold, and evaluation metrics should be documented from the actual implementation.

## 10. Technologies

* Python
* MuJoCo
* NumPy
* Matplotlib
* Git and GitHub

## 11. Project Structure

```text
Drone_MuJoCo_Project/
├── assets/
│   └── amrita_logo.png
├── analysis/
├── control/
├── experiments/
├── navigation/
├── prediction/
├── results/
├── sensors/
├── simulation/
└── README.md
```

## 12. Current Status

The project is under development. Individual simulation scenarios and control components are being developed and tested incrementally.

Only verified implementation and simulation results should be reported as completed.

## 13. Future Work

* Validate multi-obstacle navigation.
* Extend navigation to multiple targets.
* Evaluate wind stabilization and recovery.
* Integrate and evaluate safety prediction.
* Document simulation parameters and experimental results.

---

<div align="center">

**AMRITA VISHWA VIDYAPEETHAM — COIMBATORE**

**Team AB14**

</div>
