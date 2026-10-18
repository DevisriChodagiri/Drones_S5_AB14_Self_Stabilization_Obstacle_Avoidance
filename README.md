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

## TITLE

### Autonomous Self-Stabilizing Drone with Obstacle Avoidance

---
## ABSTRACT

This project presents an autonomous quadrotor that stabilizes itself and avoids obstacles, implemented in
**Python and PyBullet**. The work has two main modules.

The first module is **self-stabilization**. The drone's roll and pitch are measured, the error is computed, and a
mathematical **cascaded PID controller** (P, I and D terms computed and logged separately) converts the correction
into individual motor thrusts through a motor mixer. The controller is tested with torque disturbances and with a
circular trajectory under constant wind. A **data-driven system identification** study learns how the drone moves
from flight data: a linear model `x(k+1) = A x(k) + B u(k)` solved with the **pseudo-inverse**, and a **nonlinear**
model discovered by sparse regression (SINDy), both compared with the analytic Newton-Euler physics.

The second module is **intelligent obstacle avoidance**. The distance to an obstacle, the closing speed and the
time-to-collision are estimated and combined with roll and pitch. A **Logistic Regression** model and a **pgmpy
Bayesian network** predict the risk **SAFE / WARNING / DANGER**; an **expected-cost decision controller** selects
**CONTINUE / SLOW_DOWN / AVOID**, and the avoidance logic changes the drone's velocity.

Finally, all parts are integrated in PyBullet, including a **realistic 3-D world** with trees, buildings, a moving
bird, and **turbulent wind with gusts**, and the complete system is evaluated quantitatively.

---

## INTRODUCTION

Drones are used in surveillance, agriculture, inspection, delivery, mapping and disaster management. To operate
autonomously, a drone must keep its orientation stable and respond to its surroundings. A quadrotor is an unstable,
**nonlinear** system: its thrust direction depends on sine and cosine of roll and pitch, its rotation equations
contain products of angular rates, and aerodynamic drag grows with the square of speed. Obstacles and wind make the
problem harder.

This project combines classical control (PID), data-driven modelling (system identification), and probabilistic
machine learning (Logistic Regression, Bayesian network) into one autonomous flight framework.

**Major components**

* Mathematical PID self-stabilization (cascaded position/velocity, attitude, altitude, yaw loops)
* Quadrotor physics: motor mixer, motor lag, drag, wind
* Data-driven system identification (linear `A, B` with pseudo-inverse; nonlinear SINDy)
* Obstacle features: distance, closing speed, time-to-collision, roll, pitch
* ML risk prediction: Logistic Regression and pgmpy Bayesian network
* Expected-cost decision controller and avoidance logic
* Full PyBullet simulation and realistic world with wind

---

## METHODOLOGY

### System operation (every 1/120 s)

```
obstacles + drone state
        |
        v
 features: distance, closing speed, time-to-collision, roll, pitch
        |
        v
 risk model (Logistic Regression / pgmpy Bayesian network) --> P(SAFE), P(WARNING), P(DANGER)
        |
        v
 decision controller (minimum expected cost) --> CONTINUE / SLOW_DOWN / AVOID
        |
        v
 avoidance logic --> velocity command
        |
        v
 velocity PID --> desired roll/pitch --> attitude PID --> torques
        |
        v
 motor mixer --> 4 motor thrusts --> PyBullet physics (+ wind)
```

### Stages

1. **Physics model** (`physics_model.py`): motor mixer (`T, τx, τy, τz ↔ f1..f4`), first-order motor lag, quadratic
   drag, Newton-Euler equations; validated against PyBullet.
2. **PID stabilization** (`self_stabilizing_pid_ml.py`): `u = Kp·e + Ki·∫e dt + Kd·de/dt` with anti-windup; the error,
   integral, derivative and the P, I, D contributions are plotted individually. Experiments: attitude disturbance
   and circular trajectory under wind (P, PI, PD, PID compared).
3. **System identification (ML part)**: the drone is flown with random excitation, data `(x_k, u_k) → x_{k+1}` is
   recorded, a linear model is fitted with the pseudo-inverse `Θ = X_next · pinv([x; u; 1])`, and a nonlinear sparse
   model is fitted with a library of candidate functions (sin, cos, products, drag) and sequentially thresholded least
   squares. Models are compared on unseen data.
4. **Risk prediction** (`obstacle_avoidance_pgmpy.py`): Logistic Regression and a pgmpy Bayesian network
   (`Distance→TTC←ClosingSpeed`, `Distance, TTC, Tilt → Risk`) with CPDs learned from data and exact inference.
5. **Decision and avoidance**: `action* = argmin_a Σ_r P(r)·Cost[a][r]`; the action sets the velocity command
   (go to goal, slow down, sidestep).
6. **Integration** (`full_simulation.py`) and **realistic world with wind** (`realistic_simulation.py`).

---

## PROJECT STRUCTURE

| File | Purpose |
|---|---|
| `obstacle_avoidance_pgmpy.py` | Complete obstacle avoidance in one file (features, Logistic Regression, pgmpy network, decision, avoidance, tests) |
| `self_stabilizing_pid_ml.py` | Mathematical PID, experiments, data-driven system identification |
| `physics_model.py` | Drone physics: mixer, motor lag, drag, Newton-Euler model, PyBullet drone, validation |
| `full_simulation.py` | Full system in PyBullet: zig-zag and crossing obstacles with metrics |
| `realistic_simulation.py` | Full system in a realistic world: trees, buildings, bird, wind |
| `requirements.txt` | Python packages |
| `plots/` | Generated figures |

---

## INSTALLATION AND USAGE

```bash
pip install numpy pandas matplotlib scikit-learn pgmpy pybullet
```

```bash
python physics_model.py              # physics validation + open-loop test
python self_stabilizing_pid_ml.py    # PID experiments + system identification
python obstacle_avoidance_pgmpy.py   # risk models, zig-zag, obstacle speed sweep
python full_simulation.py            # 3-D PyBullet window (zig-zag, crossing obstacle)
python realistic_simulation.py       # 3-D PyBullet window (trees, buildings, wind)
```

Options: `--headless` (no 3-D window), `--no-show` (only save plots), `--wind-speed 6`, `--wind-dir 0`,
`--model lr|bn`, `--speed 3`, `--eval`, `--snapshot`, `--opengl2` (if the 3-D window is blank).
All figures are saved in `plots/`.

---

## RESULTS

The implemented components were tested separately and then together. Numbers below come from our runs and will vary
slightly with random seeds.

### 1. Physics validation
PyBullet and the analytic Newton-Euler model agree closely (RMSE about 1e-3 or less per state over 0.4 s). An
open-loop test with constant hover thrust shows that a small torque kick is never corrected, which shows why
feedback control is needed.

### 2. PID stabilization and trajectory tracking
* **Attitude disturbance:** roll and pitch torque kicks are rejected; the plots show `e`, `∫e`, `de/dt`, `P`, `I`, `D`
  individually.
* **Circular trajectory with constant wind (0.6 N from t = 6 s):** P-only and PI become unstable (no damping); PD is
  stable but keeps a steady offset; PID gives the smallest error.

| Position loop | RMSE (m) | Final error (m) |
|---|---|---|
| P | 2.05 | 2.82 |
| PI | 3.96 | 5.16 |
| PD | 0.36 | 0.46 |
| **PID** | **0.18** | **0.14** |

### 3. Data-driven system identification
0.5 s open-loop prediction error on unseen flight data:

| Model | roll φ (rad) | pitch θ (rad) | vx (m/s) |
|---|---|---|---|
| Linear (`A, B`, pseudo-inverse) | 0.0191 | 0.0175 | 0.1515 |
| **Nonlinear (SINDy)** | **0.0010** | **0.0005** | **0.0184** |
| Analytic physics | 0.0097 | 0.0102 | 0.0653 |

The linear model is accurate near hover but fails at larger angles; the nonlinear sparse model is about 10 times more
accurate and recovers terms such as `τx/Ixx`, trigonometric thrust terms and drag.

### 4. Risk classification (held-out data)

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.87 |
| pgmpy Bayesian network | 0.78 |

The Bayesian network is less accurate because it uses discretised features, but it gives interpretable probabilities.

### 5. Obstacle avoidance against obstacle speed (30 random crossing scenarios per speed)

| Obstacle speed (m/s) | Collision rate, no avoidance | Collision rate, with avoidance |
|---|---|---|
| 0.5 | 100 % | 0 % |
| 1.0 | 70 % | 7 % |
| 2.0 | 67 % | 0 % |
| 3.0 | 53 % | 3 % |
| 4.0 | 53 % | 13 % |

Without avoidance, the rate falls with speed partly because fast obstacles often cross before the drone arrives.

### 6. Full system in PyBullet
Zig-zag between three obstacles with no collision. In crossing-obstacle tests the altitude error stayed at or below
0.01 m during avoidance. (Small sample: 6 scenarios per speed; at 1 m/s two of six runs still collided.)

### 7. Realistic world with wind
Trees, buildings, a shed, a patrolling bird and wind (mean + turbulence + gusts). With avoidance the drone reached the
goal without collision at wind speeds of 0, 2 and 4 m/s (3 runs each) and held its altitude within about 0.03 m; at
6 m/s collisions occurred, and without avoidance every run collided.

---

## WHAT IS IMPLEMENTED, TESTED AND STILL LIMITED

**Implemented:** PID with separate P, I, D terms, motor mixer and physics, linear and nonlinear system identification,
Logistic Regression and Bayesian-network risk models, expected-cost decision, avoidance logic, full PyBullet
integration, realistic world with wind.

**Tested:** physics against analytic model; disturbance rejection; P/PI/PD/PID comparison; identification on unseen
data; classifier accuracy; obstacles at increasing speeds; wind sweep.

**Limitations and next steps**

* Risk labels come from a rule-based generator with noise, not from real flight data.
* The Bayesian network is less accurate than Logistic Regression (discretised features).
* The avoidance planner is reactive (no global planning); very fast obstacles and strong wind can still cause collisions.
* Obstacles are 2-D circles at a fixed altitude; PyBullet test sample sizes are small.
* The identified model is not yet used inside the controller (for example model-predictive control).

---

## CONCLUSION

The project combines a mathematically defined PID stabilizer, data-driven nonlinear system identification and a
probabilistic obstacle-avoidance pipeline in one PyBullet simulation. Experiments show that PID with all three terms
rejects disturbances best, that a nonlinear identified model predicts the drone's motion far better than a linear one,
and that the avoidance pipeline greatly reduces collisions in the tested scenarios. Future work will use the
identified model inside the controller, test with larger and more varied obstacle scenarios, add global path planning,
learn risk from real data, and move toward hardware implementation.

---

## REFERENCES

1. López-Sánchez, I., & Moreno-Valenzuela, J. (2023). PID Control of Quadrotor UAVs: A Survey. *Annual Reviews in Control*, 56, 100900.
2. Sun, X., & Yang, J. (2023). Design of a Quadcopter Based on Ultrasonic Obstacle Avoidance. *Journal of Hebei University of Water Resources and Electric Engineering*.
3. Lotufo, M. A., Colangelo, L., Perez-Montenegro, C. N., Canuto, E., & Novara, C. (2019). UAV Quadrotor Attitude Control: An ADRC-EMC Combined Approach. *Control Engineering Practice*, 84, 13–22.
4. López-Sánchez, I., & Moreno-Valenzuela, J. (2019). Position and Attitude Control of Multi-Rotor Aerial Vehicles: A Survey. *Annual Reviews in Control*.
5. Design of Rules for In-Flight Non-Parametric Tuning of PID Controllers for Unmanned Aerial Vehicles. (2019). *Journal of the Franklin Institute*.
6. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.
7. Ankan, A., & Panda, A. (2015). pgmpy: Probabilistic Graphical Models using Python. *Proceedings of the 14th Python in Science Conference*.
8. Coumans, E., & Bai, Y. PyBullet, a Python module for physics simulation for games, robotics and machine learning.

---

## LICENSE

This project is licensed under the MIT License.
