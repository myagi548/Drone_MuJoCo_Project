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

The project combines PID control, trajectory planning, obstacle avoidance, state-space analysis, and machine learning-based safety prediction.

## 2. Project Objectives

* Implement drone simulation using MuJoCo.
* Stabilize roll and pitch using PID control.
* Generate smooth trajectories using cubic B-splines.
* Detect obstacles and generate collision-free paths.
* Analyze system observability and controllability.
* Predict unsafe attitude conditions using Logistic Regression.
* Recover from unsafe attitude conditions.
* Evaluate flight under wind disturbances.
* Extend navigation to multiple obstacles and targets.

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

| Component             | Technology                       |
| --------------------- | -------------------------------- |
| Programming           | Python                           |
| Physics simulation    | MuJoCo                           |
| Numerical computation | NumPy                            |
| Trajectory planning   | Cubic B-splines                  |
| Flight control        | PID                              |
| Safety prediction     | Logistic Regression              |
| Visualization         | MuJoCo viewer and plotting tools |

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
|
+-- prediction/
+-- results/
+-- sensors/
|
+-- simulation/
|   +-- drone_model.xml
|   +-- mixer.py
|   +-- mujoco_force_assignment_test.py
|   +-- pid_mixer_mujoco_test.py
|   +-- pid_mixer_mujoco_viewer.py
|   +-- pid_roll_test.py
|   +-- position_bspline_mujoco_test.py
|
+-- assets/
|   +-- amrita_logo.png
|
+-- README.md
+-- .gitignore
```

## 6. Mathematical Model

### 6.1 Rotational Dynamics

The simplified rotational dynamics about one axis are:

$$
I\ddot{\theta}(t)=\tau(t)
$$

Taking the Laplace transform with zero initial conditions:

$$
Is^2\Theta(s)=\Tau(s)
$$

The transfer function is:

$$
\boxed{
G(s)=\frac{\Theta(s)}{\Tau(s)}
=\frac{1}{Is^2}
}
$$

where \(I\) is the moment of inertia, \(\theta\) is the angular position, and \(\tau\) is the applied torque.

This is a simplified single-axis model; a full drone model includes coupled rotational and translational dynamics.

### 6.2 PID Controller

The attitude error is:

$$
e(t)=\theta_d(t)-\theta(t)
$$

The PID control law is:

$$
\boxed{
u(t)=K_p e(t)
+K_i\int_0^t e(\lambda)\,d\lambda
+K_d\frac{de(t)}{dt}
}
$$

The proportional, integral, and derivative terms are:

$$
u_P(t)=K_p e(t)
$$

$$
u_I(t)=K_i\int_0^t e(\lambda)\,d\lambda
$$

$$
u_D(t)=K_d\frac{de(t)}{dt}
$$

The total controller output is:

$$
u(t)=u_P(t)+u_I(t)+u_D(t)
$$

### 6.3 PID Gain Derivation

For the plant:

$$
I\ddot{\theta}(t)=\tau(t)
$$

the transfer function is:

$$
G(s)=\frac{1}{Is^2}
$$

Using a PID controller:

$$
C(s)=K_p+\frac{K_i}{s}+K_d s
$$

The closed-loop characteristic polynomial is:

$$
Is^3+K_d s^2+K_p s+K_i=0
$$

Choose the desired characteristic polynomial:

$$
(s^2+2\zeta\omega_n s+\omega_n^2)(s+p_3)
$$

Expanding:

$$
\begin{aligned}
& s^3
+(2\zeta\omega_n+p_3)s^2\\
&+(\omega_n^2+2\zeta\omega_n p_3)s\\
&+\omega_n^2p_3
\end{aligned}
$$

Matching coefficients gives:

$$
\boxed{
K_d=I(2\zeta\omega_n+p_3)
}
$$

$$
\boxed{
K_p=I(\omega_n^2+2\zeta\omega_n p_3)
}
$$

$$
\boxed{
K_i=I\omega_n^2p_3
}
$$

where \(\zeta\) is the damping ratio, \(\omega_n\) is the natural frequency, and \(p_3\) is the additional real pole.

These formulas apply to the simplified plant and the stated controller structure. The gains must be validated against the actual MuJoCo dynamics and actuator limits.

## 7. B-Spline Trajectory Planning

### 7.1 General B-Spline Curve

A B-spline curve of degree \(p\) is:

$$
\boxed{
P(u)=\sum_{i=0}^{n}N_{i,p}(u)P_i
}
$$

where \(P_i\) are control points, \(N_{i,p}(u)\) are basis functions, and \(u\) is the curve parameter.

### 7.2 Degree-Zero Basis Function

$$
\boxed{
N_{i,0}(u)=
\begin{cases}
1, & t_i\leq u<t_{i+1},\\
0, & \text{otherwise}.
\end{cases}
}
$$

Here, \(t_i\) and \(t_{i+1}\) are consecutive knot values.

### 7.3 Cox-de Boor Recursion

For \(p>0\):

$$
\begin{aligned}
N_{i,p}(u)
={}&
\frac{u-t_i}{t_{i+p}-t_i}N_{i,p-1}(u)\\
&+
\frac{t_{i+p+1}-u}{t_{i+p+1}-t_{i+1}}
N_{i+1,p-1}(u)
\end{aligned}
$$

A term with a zero denominator is taken as zero.

### 7.4 Cubic B-Spline

For a cubic B-spline:

$$
p=3
$$

The curve is:

$$
P(u)=\sum_{i=0}^{n}N_{i,3}(u)P_i
$$

The first and second derivatives describe trajectory velocity and acceleration:

$$
V(u)=\frac{dP(u)}{du}
$$

$$
A(u)=\frac{d^2P(u)}{du^2}
$$

For time-parameterized motion \(u=u(t)\):

$$
\frac{dP}{dt}
=
\frac{dP}{du}\frac{du}{dt}
$$

The trajectory must be checked for obstacle clearance after spline construction.

## 8. Path Planning and Obstacle Avoidance

### 8.1 Straight-Line Reference Path

A straight-line path from start to goal is:

$$
\boxed{
P(\lambda)=P_s+\lambda(P_g-P_s)
}
$$

where:

$$
0\leq\lambda\leq1
$$

For \(N\) intervals:

$$
\lambda_k=\frac{k}{N},
\qquad k=0,1,\ldots,N
$$

The sampled positions are:

$$
P_k=P_s+\frac{k}{N}(P_g-P_s)
$$

### 8.2 Axis-Aligned Box Obstacle

For obstacle center \(c\) and size vector \(d\):

$$
b_{\min}=c-\frac{d}{2}
$$

$$
b_{\max}=c+\frac{d}{2}
$$

Inflating the obstacle by safety distance \(d_s\):

$$
b_{\min}^{\,\mathrm{inflated}}
=b_{\min}-d_s
$$

$$
b_{\max}^{\,\mathrm{inflated}}
=b_{\max}+d_s
$$

A candidate path is accepted only if it maintains the required clearance from the inflated obstacle.

## 9. State-Space Analysis

### 9.1 State-Space Model

A linear system is represented by:

$$
\boxed{
\dot{x}(t)=Ax(t)+Bu(t)
}
$$

$$
\boxed{
y(t)=Cx(t)+Du(t)
}
$$

where \(x\) is the state vector, \(u\) is the input, and \(y\) is the output.

### 9.2 Observability

The observability matrix is:

$$
\boxed{
\mathcal{O}=
\begin{bmatrix}
C\\
CA\\
CA^2\\
\vdots\\
CA^{n-1}
\end{bmatrix}
}
$$

The system is observable if:

$$
\boxed{
\operatorname{rank}(\mathcal{O})=n
}
$$

where \(n\) is the number of states.

### 9.3 Controllability

The controllability matrix is:

$$
\boxed{
\mathcal{C}=
\begin{bmatrix}
B & AB & A^2B & \cdots & A^{n-1}B
\end{bmatrix}
}
$$

The system is controllable if:

$$
\boxed{
\operatorname{rank}(\mathcal{C})=n
}
$$

The matrices \(A\), \(B\), and \(C\) must correspond to the actual state and measurement definitions used in the simulation.

## 10. Logistic Regression Safety Prediction

The Logistic Regression model estimates the probability of a designated safety class:

$$
\boxed{
P(y=1\mid x)=\frac{1}{1+e^{-z}}
}
$$

where:

$$
z=w^Tx+b
$$

For a feature vector with \(m\) features:

$$
z=\sum_{j=1}^{m}w_jx_j+b
$$

The predicted class is:

$$
\hat{y}=
\begin{cases}
1, & P(y=1\mid x)\geq T,\\
0, & P(y=1\mid x)<T.
\end{cases}
$$

where \(T\) is the classification threshold.

Possible input features include roll, pitch, angular velocity, and trajectory tracking error. The actual feature set, labels, threshold, and model performance must be documented after training and testing.

## 11. Recovery Control

The recovery controller is intended to bring the drone back toward its desired attitude when an unsafe condition is detected.

For roll and pitch:

$$
e_\phi(t)=\phi_d(t)-\phi(t)
$$

$$
e_\theta(t)=\theta_d(t)-\theta(t)
$$

The corresponding PID outputs are:

$$
u_\phi(t)=
K_{p,\phi}e_\phi(t)
+K_{i,\phi}\int_0^t e_\phi(\lambda)d\lambda
+K_{d,\phi}\frac{de_\phi(t)}{dt}
$$

$$
u_\theta(t)=
K_{p,\theta}e_\theta(t)
+K_{i,\theta}\int_0^t e_\theta(\lambda)d\lambda
+K_{d,\theta}\frac{de_\theta(t)}{dt}
$$

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

$$
I\ddot{\theta}(t)=\tau_c(t)+\tau_w(t)
$$

where \(\tau_c(t)\) is the controller torque and \(\tau_w(t)\) is the disturbance torque caused by wind.

The controller aims to reduce the attitude error:

$$
e(t)=\theta_d(t)-\theta(t)
$$

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

| Experiment             | Objective                                        |
| ---------------------- | ------------------------------------------------ |
| Attitude stabilization | Evaluate PID roll and pitch control              |
| Rotor force assignment | Verify force application                         |
| Position control       | Evaluate position tracking                       |
| B-spline trajectory    | Generate and track a smooth path                 |
| Single obstacle        | Test basic obstacle avoidance                    |
| Multiple obstacles     | Test path planning in a more complex environment |
| Multiple targets       | Test sequential target navigation                |
| Wind disturbance       | Evaluate stabilization and trajectory resumption |
| Safety prediction      | Evaluate unsafe-state classification             |
| Recovery controller    | Evaluate attitude recovery                       |

## 16. Results

Simulation results, plots, and performance measurements will be added after the corresponding experiments have been executed and verified.

No unverified performance values are reported here.

## 17. Team Details

**Institution:** Amrita Vishwa Vidyapeetham
**Campus:** Coimbatore
**Team:** AB14

| Name    | Student ID       | Email                                                               |
| ------- | ---------------- | ------------------------------------------------------------------- |
| Devisri | CB.SC.U4AIE24163 | [devisri7142@gmail.com](mailto:devisri7142@gmail.com)               |
| Monisha | CB.SC.U4AIE24157 | [vemurimonishareddy@gmail.com](mailto:vemurimonishareddy@gmail.com) |
| Myagi   | CB.SC.U4AIE24143 | [patimamyagi@gmail.com](mailto:patimamyagi@gmail.com)               |

## 18. Future Work

* Complete and validate the baseline simulation.
* Extend obstacle avoidance to multiple obstacles.
* Extend navigation to multiple targets.
* Implement wind stabilization and trajectory resumption.
* Integrate and evaluate the safety prediction model.
* Validate recovery behavior under disturbances.
* Record reproducible results and prepare experiment documentation.

---

<p align="center">
  <strong>Autonomous Self-Stabilizing Drone with Obstacle Avoidance</strong><br>
  Amrita Vishwa Vidyapeetham - Coimbatore
</p>
