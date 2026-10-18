<p align="center">
  <img width="441" height="114" alt="logo-branding-amrita-universiy-2024" src="https://github.com/user-attachments/assets/d939ec0b-574a-4d8e-8140-80275a9c9fa7" />
</p>

<div align="center"> <h2> AUTONOMOUS SELF-STABILIZING DRONE WITH OBSTACLE AVOIDANCE </h2> </div>

<p align="center">
  <img src="./results/amrita_logo.png" alt=""C:\Users\DEVI SRI\OneDrive\Desktop\Downloads\logo-branding-amrita-universiy-2024.jpeg"" width="200">
</p>

<p align="center">
  <b>AMRITA VISHWA VIDYAPEETHAM</b><br>
  Coimbatore, Tamil Nadu<br>
  Department of Computer Science and Engineering(Artificial Intelligence)
</p>

---

## TEAM MEMBER DETAILS

| S.No. | Name | Roll Number | Email |
|---|---|---|---|
| 1 | Devisri | CB.SC.U4AIE24163 | devisri7142@gmail.com |
| 2 | Monisha | CB.SC.U4AIE24157 | vemurimonishareddy@gmail.com |
| 3 | Myagi | CB.SC.U4AIE24143 | patimamyagi@gmail.com |

**Team Number:** AB14

---


# Autonomous Self-Stabilizing Drone with Intelligent Obstacle Avoidance

## Abstract

This project presents an autonomous drone simulation that combines quadrotor physics, self-stabilization, system identification, machine learning, Bayesian probabilistic reasoning, and intelligent obstacle avoidance. The system is designed to maintain stable flight while navigating a forest-like environment containing static obstacles, moving objects, and external disturbances such as wind.

A physics-based model represents translational and rotational motion, gravity, thrust, aerodynamic drag, and external forces. A cascaded Proportional-Integral-Derivative (PID) controller stabilizes the drone by reducing position, velocity, altitude, and attitude errors. Linear system identification and Sparse Identification of Nonlinear Dynamics (SINDy) are used to estimate and analyze drone dynamics.

For obstacle avoidance, the system extracts distance, closing speed, time to collision, roll, and pitch. Logistic Regression and a Bayesian Network estimate risk probabilities. An expected-cost decision mechanism selects an appropriate response: CONTINUE, SLOW_DOWN, or AVOID. The selected action modifies the desired velocity, which is passed to the stabilization controller.

The complete framework is integrated into PyBullet and evaluated through trajectory tracking, disturbance rejection, obstacle-crossing experiments, and forest-like navigation scenarios.

---

## 1. Introduction

Autonomous drones must maintain flight stability while responding to changing environmental conditions. A drone may be pushed away from its desired trajectory by wind, lose stability during aggressive manoeuvres, or collide with an obstacle if it cannot react in time.

This project addresses two interconnected problems:

1. **Self-stabilization:** Maintaining the desired position, velocity, altitude, and orientation through feedback control.
2. **Intelligent obstacle avoidance:** Estimating collision risk and selecting suitable movement commands to avoid obstacles.

The system combines classical control theory with machine learning and probabilistic inference. The physics model simulates the drone's response to motor commands, while the stabilization controller corrects flight errors. The obstacle-avoidance module modifies the desired movement based on environmental risk.

### 1.1 Objectives

- Simulate quadrotor translational and rotational dynamics.
- Implement cascaded PID control for stable flight.
- Compare P, PI, PD, and PID controllers.
- Evaluate disturbance rejection under wind.
- Estimate system dynamics using linear regression and SINDy.
- Extract obstacle-related features.
- Compare Logistic Regression and Bayesian Network risk prediction.
- Apply expected-cost reasoning to select avoidance actions.
- Integrate stabilization and avoidance in PyBullet.
- Evaluate trajectory tracking, collisions, clearance, and goal reaching.

---

## 2. Overall System Architecture

```text
        Environment and Obstacles
                   |
                   v
           Drone State Estimation
       Position, Velocity, Orientation
                   |
                   v
             Feature Extraction
       Distance, Closing Speed, TTC
                   |
                   v
             Risk Estimation
          /                  \
         v                    v
 Logistic Regression    Bayesian Network
          \                  /
           v                v
             Risk Probabilities
           SAFE / WARNING / DANGER
                   |
                   v
          Expected-Cost Decision
       CONTINUE / SLOW_DOWN / AVOID
                   |
                   v
          Desired Velocity Command
                   |
                   v
            Cascaded PID Control
                   |
                   v
          Desired Thrust and Torques
                   |
                   v
               Motor Mixer
                   |
                   v
             Quadrotor Physics
                   |
                   v
             Updated Drone State
                   |
                   +---- Feedback
```

### Pipeline Explanation

1. The drone's current state is obtained from the simulation.
2. Obstacle distance and relative motion are used to calculate risk-related features.
3. Machine-learning and probabilistic models estimate risk.
4. An expected-cost mechanism selects the most suitable action.
5. The avoidance module modifies the desired velocity.
6. The cascaded PID controller converts the desired movement into control commands.
7. The motor mixer converts thrust and torque requests into individual rotor commands.
8. The physics engine updates the drone's motion.
9. The new state is fed back into the next control iteration.

---

## 3. Physics Model of the Quadrotor

### 3.1 State Representation

The drone state is represented as

\[
\mathbf{x}=
\begin{bmatrix}
\mathbf{p}\\
\mathbf{v}\\
\boldsymbol{\eta}\\
\boldsymbol{\omega}
\end{bmatrix}
\]

where

- \(\mathbf{p}=[x,y,z]^T\): position.
- \(\mathbf{v}=[v_x,v_y,v_z]^T\): linear velocity.
- \(\boldsymbol{\eta}=[\phi,\theta,\psi]^T\): roll, pitch, and yaw.
- \(\boldsymbol{\omega}=[p_b,q_b,r_b]^T\): body angular velocity.

The state changes according to the forces and torques acting on the drone.

### 3.2 Translational Dynamics

Newton's second law states that the net force on an object equals its mass multiplied by its acceleration.

\[
\mathbf{F}_{net}=m\mathbf{a}
\]

For the quadrotor,

\[
m\dot{\mathbf{v}}
=
\mathbf{F}_T+m\mathbf{g}+\mathbf{F}_D+\mathbf{F}_W
\]

Therefore,

\[
\boxed{
\dot{\mathbf{v}}
=
\frac{
\mathbf{F}_T+m\mathbf{g}+\mathbf{F}_D+\mathbf{F}_W
}{m}
}
\]

where

- \(m\): drone mass.
- \(\mathbf{F}_T\): total rotor thrust expressed in world coordinates.
- \(\mathbf{g}\): gravitational acceleration.
- \(\mathbf{F}_D\): aerodynamic drag.
- \(\mathbf{F}_W\): external wind force.

**Why it is used:** This equation determines how the drone accelerates due to thrust, gravity, drag, and wind.

### 3.3 Thrust and Orientation

The rotors generate thrust along the drone's body vertical axis. The thrust direction in world coordinates depends on the drone's orientation.

\[
\mathbf{F}_T
=
R
\begin{bmatrix}
0\\
0\\
T
\end{bmatrix}
\]

where \(R\) is the rotation matrix and \(T\) is total thrust.

For a common roll-pitch-yaw convention,

\[
R=R_z(\psi)R_y(\theta)R_x(\phi).
\]

The individual rotation matrices are

\[
R_x(\phi)=
\begin{bmatrix}
1&0&0\\
0&\cos\phi&-\sin\phi\\
0&\sin\phi&\cos\phi
\end{bmatrix}
\]

\[
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{bmatrix}
\]

\[
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
\]

**Why it is used:** Tilting the drone changes the direction of thrust, allowing horizontal movement.

### 3.4 Gravity and Hovering

When the drone hovers at a constant altitude without wind or acceleration,

\[
T=mg.
\]

If \(T>mg\), the drone accelerates upward when its thrust is vertical.

If \(T<mg\), the drone accelerates downward.

**Why it is used:** The hovering condition provides the reference thrust required to balance gravity.

### 3.5 Aerodynamic Drag

A simplified quadratic drag model is

\[
\boxed{
\mathbf{F}_D=-c_D\|\mathbf{v}\|\mathbf{v}
}
\]

where \(c_D\) is the drag coefficient.

The drag force opposes motion and becomes stronger as speed increases.

**Why it is used:** Drag prevents unrealistic motion by representing aerodynamic resistance.

### 3.6 Wind Disturbance

Wind is represented as an external force:

\[
\mathbf{F}_W=
\begin{bmatrix}
F_{wx}\\
F_{wy}\\
F_{wz}
\end{bmatrix}.
\]

The corresponding acceleration is

\[
\mathbf{a}_W=\frac{\mathbf{F}_W}{m}.
\]

**Why it is used:** Wind disturbances help evaluate whether the stabilization controller can maintain the desired trajectory.

### 3.7 Rotational Dynamics

The rotational dynamics of a rigid body are described by Euler's equation:

\[
\boxed{
\boldsymbol{\tau}
=
I\dot{\boldsymbol{\omega}}
+
\boldsymbol{\omega}\times(I\boldsymbol{\omega})
}
\]

Rearranging,

\[
\boxed{
\dot{\boldsymbol{\omega}}
=
I^{-1}
\left[
\boldsymbol{\tau}
-
\boldsymbol{\omega}\times(I\boldsymbol{\omega})
\right]
}
\]

where

- \(\boldsymbol{\tau}\): applied torque.
- \(I\): moment-of-inertia matrix.
- \(\boldsymbol{\omega}\): body angular velocity.
- \(\dot{\boldsymbol{\omega}}\): angular acceleration.

**Why it is used:** It describes how rotor-generated torques change the drone's roll, pitch, and yaw rates.

### 3.8 Rotor Thrust and Motor Mixing

The thrust generated by an individual rotor is approximated by

\[
T_i=k_f\Omega_i^2
\]

where \(k_f\) is the thrust coefficient and \(\Omega_i\) is rotor angular speed.

The total thrust is

\[
T=T_1+T_2+T_3+T_4.
\]

The motor mixer maps total thrust and desired torques to individual rotor thrusts:

\[
\begin{bmatrix}
T\\
\tau_x\\
\tau_y\\
\tau_z
\end{bmatrix}
=
M
\begin{bmatrix}
T_1\\
T_2\\
T_3\\
T_4
\end{bmatrix}.
\]

The mixing matrix \(M\) depends on the quadrotor geometry and rotor configuration.

**Why it is used:** The controller calculates desired thrust and torques, whereas the physics simulation requires individual rotor commands.

---

## 4. Self-Stabilization Using PID Control

### 4.1 What Is Self-Stabilization?

Self-stabilization is the ability of the drone to detect deviations from its desired state and generate corrective actions.

For example, if wind pushes the drone away from its target, the controller measures the error and adjusts thrust and orientation to bring the drone back toward the desired trajectory.

### 4.2 Error Calculation

The control error is

\[
\boxed{e(t)=r(t)-y(t)}
\]

where \(r(t)\) is the desired reference and \(y(t)\) is the measured value.

For altitude,

\[
e_z=z_{\mathrm{desired}}-z_{\mathrm{actual}}.
\]

A positive altitude error means the drone is below its desired altitude.

### 4.3 PID Equation

The PID control law is

\[
\boxed{
u(t)=K_Pe(t)
+K_I\int_0^t e(\tau)\,d\tau
+K_D\frac{de(t)}{dt}
}
\]

where

- \(K_P\): proportional gain.
- \(K_I\): integral gain.
- \(K_D\): derivative gain.
- \(e(t)\): control error.
- \(u(t)\): control output.

### 4.4 Proportional Control

\[
P(t)=K_Pe(t)
\]

The proportional term responds to the current error.

A larger error produces a larger correction. Excessive proportional gain can cause oscillation or overshoot.

### 4.5 Integral Control

\[
I(t)=K_I\int_0^t e(\tau)\,d\tau
\]

In discrete time,

\[
S_k=S_{k-1}+e_k\Delta t
\]

and

\[
I_k=K_I S_k.
\]

The integral term accumulates past error and can eliminate persistent offsets. Excessive integral accumulation can cause windup, so the accumulated value should be limited.

### 4.6 Derivative Control

\[
D(t)=K_D\frac{de(t)}{dt}
\]

The discrete derivative estimate is

\[
\dot e_k\approx\frac{e_k-e_{k-1}}{\Delta t}.
\]

The derivative term responds to how rapidly the error changes and can reduce oscillation.

Because differentiation amplifies measurement noise, filtering may be applied:

\[
d_k=d_{k-1}+\alpha(d_{\mathrm{raw},k}-d_{k-1})
\]

where \(0<\alpha\leq1\).

### 4.7 Anti-Windup

The integral contribution can be limited using

\[
|K_I S_k|\leq I_{\max}.
\]

For \(K_I>0\),

\[
S_k=
\operatorname{clip}
\left(
S_{k-1}+e_k\Delta t,
-\frac{I_{\max}}{K_I},
\frac{I_{\max}}{K_I}
\right).
\]

**Why it is used:** It prevents excessive integral accumulation when the motors cannot provide the requested control output.

### 4.8 Cascaded PID Architecture

```text
Desired Position
       |
       v
Position Controller
       |
       v
Desired Velocity
       |
       v
Velocity Controller
       |
       v
Desired Roll and Pitch
       |
       v
Attitude Controller
       |
       v
Desired Torque and Thrust
       |
       v
Motor Mixer
       |
       v
Rotor Commands
```

Altitude and yaw are controlled through their corresponding feedback loops.

The outer loops generate movement references, while the inner attitude loop tracks the desired orientation.

**Why cascaded control is used:** Position and attitude have different dynamics. Separating their control loops helps the drone respond quickly to orientation errors while tracking slower position commands.

### 4.9 Converting Acceleration into Desired Attitude

For small angles and near-level flight,

\[
a_x\approx g\theta,
\qquad
a_y\approx-g\phi
\]

under a common coordinate convention.

Thus,

\[
\theta_{\mathrm{desired}}\approx\frac{a_x}{g},
\qquad
\phi_{\mathrm{desired}}\approx-\frac{a_y}{g}.
\]

The signs depend on the simulator's coordinate convention.

**Why it is used:** A quadrotor generates horizontal acceleration by tilting its thrust vector. The controller therefore converts horizontal movement demands into desired roll and pitch.

### 4.10 Reported PID Results

The reported position-control experiment under a circular trajectory and constant wind produced the following results:

| Controller | RMSE (m) | Final Error (m) |
|---|---:|---:|
| P | 2.05 | 2.82 |
| PI | 3.96 | 5.16 |
| PD | 0.36 | 0.46 |
| **PID** | **0.18** | **0.14** |

The full PID configuration produced the lowest reported RMSE and final error in this experiment.

These results demonstrate the importance of gain tuning. Adding an integral term does not always improve performance when the controller gains are unsuitable for the system.

---

## 5. System Identification

### 5.1 Purpose

System identification estimates the mathematical behaviour of a system using recorded inputs and outputs.

For the drone, state transitions are collected in the form

\[
(\mathbf{x}_k,\mathbf{u}_k)\rightarrow\mathbf{x}_{k+1}.
\]

The resulting data can be used to estimate a model that predicts future states.

### 5.2 Linear System Identification

A discrete-time linear model is

\[
\boxed{
\mathbf{x}_{k+1}=A\mathbf{x}_k+B\mathbf{u}_k+\mathbf{c}
}
\]

where \(A\) describes state dynamics, \(B\) describes control-input effects, and \(\mathbf{c}\) is an offset.

Using the data matrix

\[
Z=
\begin{bmatrix}
\mathbf{x}_1&\cdots&\mathbf{x}_{N-1}\\
\mathbf{u}_1&\cdots&\mathbf{u}_{N-1}\\
1&\cdots&1
\end{bmatrix}
\]

and next-state matrix

\[
X_+=
\begin{bmatrix}
\mathbf{x}_2&\cdots&\mathbf{x}_N
\end{bmatrix},
\]

the least-squares estimate is

\[
\boxed{\hat{\Theta}=X_+Z^+}
\]

where \(Z^+\) is the Moore-Penrose pseudoinverse.

**Why the pseudoinverse is used:** It estimates model parameters when the data matrix is rectangular or does not have a standard inverse.

### 5.3 Nonlinear System Identification Using SINDy

Quadrotor motion is nonlinear because of rotation, thrust direction, drag, and coupling between axes.

SINDy represents the dynamics as

\[
\boxed{
\dot{\mathbf{x}}=\Theta(\mathbf{x},\mathbf{u})\Xi
}
\]

where \(\Theta\) contains candidate functions and \(\Xi\) contains the identified coefficients.

Candidate functions may include

\[
1,\quad x_i,\quad u_j,\quad
\sin\phi,\quad\cos\phi,\quad
x_ix_j,\quad\|\mathbf{v}\|v_x.
\]

A sparse-regression objective can be written as

\[
\min_{\Xi}
\left\|
\dot X-\Theta(X,U)\Xi
\right\|_2^2
+\lambda\|\Xi\|_0.
\]

The sparsity term encourages a model with fewer active terms.

**Why SINDy is used:** It can identify interpretable nonlinear relationships from data and may represent complex drone dynamics more accurately than a simple linear model.

### 5.4 Reported Identification Results

| Model | Roll Error (rad) | Pitch Error (rad) | Horizontal Velocity Error (m/s) |
|---|---:|---:|---:|
| Linear model | 0.0191 | 0.0175 | 0.1515 |
| **Nonlinear SINDy** | **0.0010** | **0.0005** | **0.0184** |
| Analytic physics | 0.0097 | 0.0102 | 0.0653 |

For these reported tests, the nonlinear identified model produced the lowest prediction errors for roll, pitch, and horizontal velocity.

The identified model is used for analysis and prediction; it is not integrated into a model-predictive controller in the described system.

---

## 6. Intelligent Obstacle Avoidance

### 6.1 Objective

Obstacle avoidance allows the drone to respond to environmental hazards instead of blindly following its target trajectory.

The system estimates risk using obstacle distance, relative motion, time to collision, and drone tilt.

Three risk categories are considered:

- **SAFE:** Continue toward the target.
- **WARNING:** Reduce speed or increase caution.
- **DANGER:** Take a stronger avoidance action.

The available actions are CONTINUE, SLOW_DOWN, and AVOID.

### 6.2 Obstacle Distance

Let the drone position be

\[
\mathbf{p}_d=
\begin{bmatrix}
x_d\\y_d
\end{bmatrix}
\]

and obstacle position be

\[
\mathbf{p}_o=
\begin{bmatrix}
x_o\\y_o
\end{bmatrix}.
\]

The centre-to-centre distance is

\[
\boxed{
d=\|\mathbf{p}_o-\mathbf{p}_d\|_2
}
\]

or

\[
d=\sqrt{(x_o-x_d)^2+(y_o-y_d)^2}.
\]

If the obstacle and drone radii are \(r_o\) and \(r_d\), the surface clearance is

\[
\boxed{c=d-r_o-r_d}.
\]

Negative clearance indicates overlap under this simplified geometry.

### 6.3 Relative Velocity and Closing Speed

Relative velocity is

\[
\mathbf{v}_{rel}=\mathbf{v}_o-\mathbf{v}_d.
\]

The line-of-sight unit vector is

\[
\hat{\mathbf r}
=
\frac{\mathbf p_o-\mathbf p_d}
{\|\mathbf p_o-\mathbf p_d\|}.
\]

The rate of change of distance is

\[
\dot d=\hat{\mathbf r}^{T}(\mathbf v_o-\mathbf v_d).
\]

Defining positive closing speed as decreasing separation,

\[
\boxed{
v_c=-\dot d
}
\]

A positive \(v_c\) indicates that the obstacle is approaching along the line of sight.

### 6.4 Time to Collision

A simple time-to-collision approximation is

\[
\boxed{
TTC\approx\frac{d}{v_c}
}
\]

when \(v_c>0\).

This approximation assumes nearly constant closing speed. It is not reliable when the obstacle is moving away or the trajectories are changing significantly.

A more general constant-relative-velocity estimate of time to closest approach is

\[
t^*
=
-\frac{
(\mathbf p_o-\mathbf p_d)^T
(\mathbf v_o-\mathbf v_d)
}{
\|\mathbf v_o-\mathbf v_d\|^2
}
\]

when relative velocity is nonzero.

**Why TTC is used:** It combines proximity and relative motion to estimate how urgently the drone should react.

### 6.5 Feature Vector

The risk-classification feature vector is

\[
\boxed{
\mathbf{x}=
\begin{bmatrix}
d&v_c&TTC&\phi&\theta
\end{bmatrix}^{T}
}
\]

where \(d\) is distance, \(v_c\) is closing speed, \(TTC\) is time to collision, and \(\phi,\theta\) are roll and pitch.

These features help the model estimate the risk of collision.

---

## 7. Logistic Regression

### 7.1 Overview

Logistic Regression estimates class probabilities from input features.

For binary classification,

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

and

\[
P(y=1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b).
\]

For three risk categories, the multiclass softmax formulation is

\[
z_k=\mathbf{w}_k^T\mathbf{x}+b_k
\]

and

\[
\boxed{
P(y=k\mid\mathbf{x})
=
\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
}
\]

where \(K=3\) for SAFE, WARNING, and DANGER.

The probabilities satisfy

\[
\sum_{k=1}^{K}P(y=k\mid\mathbf{x})=1.
\]

### 7.2 Training Objective

The multiclass cross-entropy loss is

\[
\boxed{
L=
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{K}
y_{ik}\log p_{ik}
}
\]

where \(y_{ik}\) is the target indicator and \(p_{ik}\) is the predicted probability.

Training adjusts the model parameters to reduce the loss.

### 7.3 Why Logistic Regression Is Used

- It is computationally efficient.
- It provides class probabilities.
- It serves as a baseline for comparison.
- Its probabilities can be used by the expected-cost controller.

The reported risk labels were generated synthetically using rules and noise. Therefore, the reported accuracy measures performance on the constructed dataset rather than proving real-world flight safety.

---

## 8. Bayesian Network

### 8.1 Overview

A Bayesian Network represents probabilistic dependencies using a directed acyclic graph.

A simplified dependency structure is

```text
Distance -------> TTC <------- Closing Speed
    |               |
    |               v
    +------------> Risk <------ Tilt
```

The graph represents conditional relationships between distance, closing speed, time to collision, tilt, and risk.

### 8.2 Joint Probability Factorization

For variables \(X_1,\ldots,X_n\),

\[
\boxed{
P(X_1,\ldots,X_n)
=
\prod_{i=1}^{n}P(X_i\mid Pa(X_i))
}
\]

where \(Pa(X_i)\) denotes the parent variables of \(X_i\).

For the simplified graph, one possible factorization is

\[
\begin{aligned}
P(D,C,T,\Theta,R)
={}&P(D)P(C)\\
&\cdot P(T\mid D,C)\\
&\cdot P(\Theta)\\
&\cdot P(R\mid D,T,\Theta).
\end{aligned}
\]

Here, \(D\) represents distance, \(C\) closing speed, \(T\) time to collision, \(\Theta\) tilt, and \(R\) risk.

The factorization must match the graph used in the actual implementation.

### 8.3 Conditional Probability Tables

A Conditional Probability Table defines the probability of a variable for each possible combination of its parent states.

For risk,

\[
P(R\mid D,T,\Theta)
\]

represents the conditional distribution of SAFE, WARNING, and DANGER.

For every parent-state combination,

\[
\sum_r P(R=r\mid D,T,\Theta)=1.
\]

### 8.4 Bayesian Inference

Given evidence \(E\), the posterior risk probability is

\[
\boxed{
P(R\mid E)
=
\frac{P(E\mid R)P(R)}{P(E)}
}
\]

for a single target variable and evidence set.

For a network with hidden variables, inference combines the conditional distributions and sums over unobserved states.

### 8.5 Why a Bayesian Network Is Used

- It explicitly represents conditional dependencies.
- It supports probabilistic inference.
- It combines multiple sources of evidence.
- It produces interpretable posterior probabilities.

The network uses discretized features, which simplify probability-table construction but may lose information from continuous measurements.

### 8.6 Model Comparison

| Aspect | Logistic Regression | Bayesian Network |
|---|---|---|
| Representation | Learned weights | Graph and conditional probability tables |
| Input | Feature vector | Variables and evidence |
| Output | Class probabilities | Posterior probabilities |
| Interpretability | Coefficients | Dependencies and CPTs |
| Main limitation | Depends on feature representation | Depends on graph structure and discretization |

Neither method is guaranteed to outperform the other in all scenarios.

---

## 9. Expected-Cost Decision-Making

### 9.1 Motivation

A risk prediction does not automatically determine the best action. The controller must consider the consequences of continuing, slowing down, or avoiding.

Let \(a\) represent an action and \(r\) represent a risk state.

The expected cost of an action is

\[
\boxed{
EC(a)=\sum_r P(r\mid\mathbf{x})C(a,r)
}
\]

where \(C(a,r)\) is the cost of taking action \(a\) under risk state \(r\).

The selected action is

\[
\boxed{
a^*=\arg\min_{a\in\mathcal A}EC(a)
}
\]

with

\[
\mathcal A=
\{\text{CONTINUE},\text{SLOW\_DOWN},\text{AVOID}\}.
\]

### 9.2 Conceptual Cost Matrix

| Action | SAFE | WARNING | DANGER |
|---|---|---|---|
| CONTINUE | Low cost | Medium cost | Very high cost |
| SLOW_DOWN | Low or medium cost | Low cost | High cost |
| AVOID | Manoeuvring cost | Low cost | Lower collision-related cost |

The exact numerical costs are implementation parameters.

**Why expected cost is used:** It considers both the probability of a risky situation and the potential consequences of each action.

---

## 10. Velocity-Based Avoidance

### 10.1 Goal-Directed Velocity

Let the target position be \(\mathbf p_g\) and drone position be \(\mathbf p_d\).

The normalized direction to the target is

\[
\hat{\mathbf g}
=
\frac{\mathbf p_g-\mathbf p_d}
{\|\mathbf p_g-\mathbf p_d\|}.
\]

The goal-directed velocity is

\[
\mathbf v_{\mathrm{goal}}=v_{\mathrm{goal}}\hat{\mathbf g}.
\]

### 10.2 Avoidance Command

A conceptual combination of goal-seeking and avoidance velocity is

\[
\boxed{
\mathbf v_{\mathrm{cmd}}
=
\mathbf v_{\mathrm{goal}}
+\lambda\mathbf v_{\mathrm{avoid}}
}
\]

where \(\lambda\) controls the influence of avoidance.

For a relative obstacle position

\[
\mathbf r=
\begin{bmatrix}
r_x\\r_y
\end{bmatrix},
\]

a possible perpendicular direction is

\[
\mathbf n=
\frac{1}{\|\mathbf r\|}
\begin{bmatrix}
-r_y\\r_x
\end{bmatrix}.
\]

The direction can be selected according to the available free space and the desired sidestep direction. This equation illustrates a possible geometric avoidance direction; the exact velocity command depends on the implemented avoidance rule.

### 10.3 Available Actions

**CONTINUE:** Follow the goal-directed velocity when the risk is sufficiently low.

**SLOW_DOWN:** Reduce the desired speed:

\[
\mathbf v_{\mathrm{cmd}}
=
\alpha\mathbf v_{\mathrm{goal}},
\qquad 0\leq\alpha<1.
\]

**AVOID:** Change the desired velocity to increase separation from the obstacle while retaining progress toward the target where possible.

### 10.4 Hysteresis

To reduce rapid switching between actions, separate thresholds can be used.

Enter avoidance when

\[
q\geq q_{\mathrm{enter}}
\]

and leave avoidance only when

\[
q\leq q_{\mathrm{exit}},
\]

where

\[
q_{\mathrm{exit}}<q_{\mathrm{enter}}.
\]

Hysteresis helps prevent repeated switching when the estimated risk fluctuates near a threshold.

### 10.5 Integration with Stabilization

The obstacle-avoidance module does not replace the flight controller. It modifies the desired velocity, which the stabilization system attempts to follow.

1. Estimate risk.
2. Select an action.
3. Generate a velocity command.
4. Calculate control errors.
5. Generate desired attitude, thrust, and torques.
6. Convert the commands into rotor thrusts.
7. Update the physics simulation.
8. Repeat using the new drone state.

---

## 11. Simulation and Evaluation

### 11.1 Simulation Environment

The system is integrated into PyBullet to simulate rigid-body motion and collision interactions.

The evaluated scenarios include:

- Zig-zag navigation around obstacles.
- Moving obstacles crossing the drone's path.
- Wind disturbances.
- A forest-like environment with trees and buildings.
- A moving bird and additional environmental disturbances.

### 11.2 Evaluation Metrics

**Position RMSE**

\[
RMSE_p=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\|\mathbf p_i-\mathbf p_{\mathrm{ref},i}\|^2
}
\]

This measures deviation from the reference trajectory.

**Collision rate**

\[
\boxed{
R_c=
\frac{N_{\mathrm{collisions}}}{N_{\mathrm{trials}}}
}
\]

Multiplying by 100 expresses the rate as a percentage.

**Goal-reaching rate**

\[
R_g=
\frac{N_{\mathrm{successful\ trials}}}{N_{\mathrm{trials}}}.
\]

**Minimum clearance**

\[
c_{\min}
=
\min_t(d(t)-r_o-r_d).
\]

**Classification accuracy**

\[
Accuracy=
\frac{\text{Correct Predictions}}{\text{Total Predictions}}.
\]

**Precision**

\[
Precision=\frac{TP}{TP+FP}.
\]

**Recall**

\[
Recall=\frac{TP}{TP+FN}.
\]

**F1-score**

\[
F1=
2\frac{Precision\cdot Recall}{Precision+Recall}.
\]

Recall for the DANGER class is important because failing to detect a dangerous situation can result in a collision.

---

## 12. Experimental Results

The following values are the reported results from the project material and apply to the specific tested configurations.

### 12.1 Physics Validation

The analytic Newton-Euler model was compared against PyBullet over a short simulation interval. The reported state RMSE was approximately \(10^{-3}\) or less per state over 0.4 seconds in the validation setup.

An open-loop test with constant hover thrust showed that an applied torque disturbance was not automatically corrected. This illustrates why feedback control is necessary for self-stabilization.

### 12.2 PID Comparison

| Controller | RMSE (m) | Final Error (m) |
|---|---:|---:|
| P | 2.05 | 2.82 |
| PI | 3.96 | 5.16 |
| PD | 0.36 | 0.46 |
| **PID** | **0.18** | **0.14** |

The full PID configuration achieved the lowest reported RMSE and final error in the tested circular-trajectory experiment under constant wind.

### 12.3 System Identification

| Model | Roll Error (rad) | Pitch Error (rad) | Horizontal Velocity Error (m/s) |
|---|---:|---:|---:|
| Linear model | 0.0191 | 0.0175 | 0.1515 |
| **Nonlinear SINDy** | **0.0010** | **0.0005** | **0.0184** |
| Analytic physics | 0.0097 | 0.0102 | 0.0653 |

The nonlinear identified model produced the lowest reported errors for the listed variables.

### 12.4 Risk Classification

| Model | Reported Accuracy |
|---|---:|
| Logistic Regression | 0.87 |
| Bayesian Network | 0.78 |

Logistic Regression achieved higher accuracy in the reported experiment. The Bayesian Network offers explicit probabilistic dependencies and interpretable posterior probabilities.

### 12.5 Obstacle-Speed Evaluation

The reported obstacle-speed experiment used 30 random crossing scenarios per speed.

| Obstacle Speed (m/s) | Collision Rate Without Avoidance | Collision Rate With Avoidance |
|---|---:|---:|
| 0.5 | 100% | 0% |
| 1.0 | 70% | 7% |
| 2.0 | 67% | 0% |
| 3.0 | 53% | 3% |
| 4.0 | 53% | 13% |

Avoidance substantially reduced the reported collision rate in these tests. However, collisions still occurred in some scenarios, especially at higher obstacle speeds.

The collision rate without avoidance is not monotonic because a faster obstacle may cross the drone's path before the drone reaches the crossing point.

### 12.6 Full-System Simulation

In the reported zig-zag scenario, the drone navigated between three obstacles without collision.

In the crossing-obstacle evaluation, altitude error remained at or below approximately 0.01 m during avoidance. The reported full-system crossing tests used six scenarios per speed; two of six runs at 1 m/s resulted in collisions.

These results demonstrate why full-system navigation must be evaluated separately from risk-classification accuracy.

### 12.7 Wind Evaluation

The reported forest-like simulation included trees, buildings, a shed, a moving bird, and wind with mean, turbulent, and gust components.

| Wind Speed | Reported Outcome With Avoidance |
|---|---|
| 0 m/s | Goal reached without collision in 3 reported runs |
| 2 m/s | Goal reached without collision in 3 reported runs |
| 4 m/s | Goal reached without collision in 3 reported runs |
| 6 m/s | Collisions occurred |

The drone maintained altitude within approximately 0.03 m in the reported successful lower-wind trials. Without avoidance, every run in the reported wind evaluation collided.

The small number of trials means that these results should be treated as preliminary simulation evidence rather than a guarantee of safe operation.

---

## 13. Limitations

### 13.1 Synthetic Risk Labels

The risk labels are generated using a rule-based process with noise. The classifier results therefore do not establish performance on real-world drone data.

### 13.2 Bayesian Network Discretization

Discretizing continuous features simplifies CPT construction but may discard useful information.

### 13.3 Reactive Avoidance

The controller responds to the current situation instead of solving a complete global trajectory-planning problem. Fast-moving obstacles and narrow passages can still cause failures.

### 13.4 Simplified Geometry

Some collision-risk calculations use simplified two-dimensional circular geometry and do not capture every three-dimensional collision possibility.

### 13.5 Limited Number of Trials

Several full-system tests use a small sample size. Larger experiments are needed to measure reliability and statistical variation.

### 13.6 Simulation-to-Reality Gap

Simulation cannot perfectly reproduce sensor noise, actuator imperfections, battery behaviour, communication delays, and every aerodynamic effect. Successful simulation does not prove that the system is ready for real-world flight.

---

## 14. Future Enhancements

1. Train the risk models on real or higher-fidelity sensor data.
2. Improve Bayesian Network structure and probability estimation.
3. Add camera-based or depth-based obstacle perception.
4. Integrate the identified nonlinear model into model-predictive control.
5. Develop multi-obstacle trajectory planning.
6. Increase the number and diversity of simulation trials.
7. Test more wind conditions and moving-obstacle trajectories.
8. Improve uncertainty estimation and risk thresholds.
9. Evaluate the system using hardware-in-the-loop testing.
10. Report statistical variation and confidence intervals.

---

## 15. Conclusion

This project integrates quadrotor physics, cascaded PID control, system identification, machine learning, Bayesian inference, and reactive obstacle avoidance into an autonomous navigation simulation.

The physics model represents translational and rotational motion under thrust, gravity, drag, and wind. The cascaded PID controller reduces position, velocity, altitude, and attitude errors. Linear pseudoinverse estimation and nonlinear sparse identification provide complementary methods for analysing drone dynamics.

The obstacle-avoidance module uses distance, closing speed, time to collision, and tilt to estimate risk. Logistic Regression and a Bayesian Network provide risk probabilities, while expected-cost reasoning selects CONTINUE, SLOW_DOWN, or AVOID. The resulting velocity command is passed to the stabilization controller so that avoidance and flight stability operate together.

The reported experiments demonstrate improved trajectory tracking with the full PID controller, lower prediction errors for the nonlinear identified model in the tested variables, and reduced collision rates when obstacle avoidance is enabled. However, fast-moving obstacles and strong wind can still cause failures.

The project provides a foundation for further work on robust autonomous navigation, probabilistic risk estimation, predictive control, and safe drone operation.

---

## References

1. López-Sánchez, I., & Moreno-Valenzuela, J. (2023). *PID Control of Quadrotor UAVs: A Survey*. Annual Reviews in Control, 56, 100900.

2. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences, 113(15), 3932–3937. https://doi.org/10.1073/pnas.1517384113

3. Ankan, A., & Panda, A. (2015). *pgmpy: Probabilistic Graphical Models using Python*. Proceedings of the 14th Python in Science Conference. https://doi.org/10.25080/Majora-7b98e3ed-007

4. Coumans, E., & Bai, Y. *PyBullet: Python Module for Physics Simulation for Games, Robotics and Machine Learning*. https://pybullet.org/
