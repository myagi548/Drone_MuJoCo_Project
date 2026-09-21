



Project Title



Autonomous Self-Stabilizing Drone with Obstacle Avoidance



1. Introduction



Autonomous drone navigation requires the ability to follow a planned path, avoid obstacles, maintain stable flight, and recover from unsafe attitude conditions. This project proposes a simulated autonomous drone system that combines trajectory planning, feedback control, obstacle avoidance, and machine learning-based safety prediction.



The system will be developed and evaluated using the MuJoCo physics simulator.



2. Problem Statement



A drone navigating through an environment containing multiple obstacles may deviate from its desired trajectory due to control errors or external disturbances such as wind. These deviations can lead to collisions or unsafe roll and pitch angles.



The project aims to develop and evaluate a simulation framework that allows a drone to navigate toward target locations while avoiding obstacles, maintaining attitude stability, and responding to potentially unsafe flight conditions.



3. Objectives



\* Develop a drone simulation environment using MuJoCo.

\* Generate collision-aware navigation paths using B-spline trajectory planning.

\* Implement PID-based attitude stabilization and position control.

\* Simulate obstacle detection and avoidance in environments containing multiple obstacles.

\* Evaluate drone navigation under both no-wind and wind-disturbance conditions.

\* Develop a Logistic Regression-based safety prediction model to identify potentially unsafe attitude conditions.

\* Investigate recovery control for reducing roll and pitch deviations and returning the drone to a stable state.

\* Evaluate the system using trajectory tracking error, target-reaching performance, obstacle collisions, and attitude response.



&#x20;4. Proposed Methodology



The project will be developed in the following stages:



1\. Simulation Setup: Create the drone model and simulation environment in MuJoCo.

2\. Trajectory Planning: Generate a collision-aware route using obstacle planning and B-spline trajectory generation.

3\. Control System: Implement PID-based control for position and attitude stabilization.

4\. Obstacle Avoidance: Detect obstacles and plan a safe route around them.

5\. Wind Simulation: Introduce external wind disturbances and evaluate the drone's ability to maintain stability and continue navigation.

6\. Safety Prediction: Use Logistic Regression to classify flight states as safe or potentially unsafe based on selected attitude-related features.

7\. Recovery Control: Investigate a recovery controller that brings roll and pitch toward their desired stable values.

8\. Testing and Evaluation: Compare simulation behavior under different scenarios and record performance metrics.



5. Tools and Technologies



\* Python

\* MuJoCo

\* NumPy

\* Matplotlib

\* Scikit-learn

\* Git and GitHub



6. Expected Outcomes



\* A simulated autonomous drone capable of following planned waypoints.

\* A trajectory-planning module designed to avoid multiple obstacles.

\* PID-based position and attitude control.

\* Separate wind and no-wind simulation scenarios.

\* A machine learning safety-classification module and recovery-control demonstration, subject to successful implementation and testing.

\* Recorded simulation results and performance comparisons.



7. Evaluation Metrics



\* Final target position error

\* Trajectory tracking error

\* Number of obstacle collisions

\* Roll and pitch deviation

\* Stabilization and recovery time

\* Performance comparison between wind and no-wind scenarios



8. Scope and Limitations



This project focuses on simulation-based development and evaluation. Results obtained in MuJoCo will not, by themselves, establish that the system is safe or ready for real-world drone deployment. Each proposed feature will be evaluated based on its actual implementation and test results.



9. Conclusion



The proposed project will investigate an integrated approach to autonomous drone navigation by combining trajectory planning, PID control, obstacle avoidance, wind-disturbance testing, and machine learning-based safety prediction in a MuJoCo simulation environment.



