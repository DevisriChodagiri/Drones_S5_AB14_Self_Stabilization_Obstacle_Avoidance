<p align="center">
  <img width="441" height="114" alt="logo-branding-amrita-universiy-2024" src="https://github.com/user-attachments/assets/d939ec0b-574a-4d8e-8140-80275a9c9fa7" />
</p>

<div align="center"> <h2> AUTONOMOUS SELF-STABILIZING DRONE WITH OBSTACLE AVOIDANCE </h2> </div>

<p align="center">
  <img src="./results/amrita_logo.png" alt="Amrita Vishwa Vidyapeetham logo" width="200">
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
## ABSTRACT

This project builds an autonomous quadrotor that stabilizes itself and avoids obstacles, implemented in **Python and
PyBullet**. It has two modules. **Self-stabilization:** a mathematical cascaded PID controller (P, I and D terms
computed and logged separately) turns roll, pitch, velocity and altitude errors into individual motor thrusts; a
data-driven system identification study learns the drone's dynamics from flight data, first as a linear model
`x(k+1) = A x(k) + B u(k)` using the pseudo-inverse and then as a nonlinear sparse model (SINDy).
**Intelligent obstacle avoidance:** distance, closing speed and time-to-collision are combined with roll and pitch and
fed to a Logistic Regression model and a pgmpy Bayesian network that predict SAFE / WARNING / DANGER; a decision
controller selects CONTINUE / SLOW_DOWN / AVOID and the avoidance logic changes the drone's velocity. Everything is
integrated in PyBullet, including a realistic world with trees, buildings, a moving bird and turbulent wind.

---

## METHODOLOGY (STEP BY STEP)

### Step 0 — Overall pipeline

```
obstacles + drone state
   -> features (distance, closing speed, time-to-collision, roll, pitch)
   -> risk model (Logistic Regression / pgmpy Bayesian network): P(SAFE), P(WARNING), P(DANGER)
   -> decision controller (minimum expected cost): CONTINUE / SLOW_DOWN / AVOID
   -> avoidance logic: velocity command
   -> velocity PID -> desired roll/pitch -> attitude PID -> torques      (self-stabilization)
   -> motor mixer -> 4 motor thrusts -> motor lag -> PyBullet physics (+ wind)
   -> drone state fed back
```

Physics runs at 240 Hz, the controller at 120 Hz.

---

### Step 1 — Physics of the drone (`physics_model.py`)

This section explains how motor commands become forces, torques, translation, and rotation. The model uses a rigid-body
quadrotor approximation. Unless otherwise stated, rotor positions and rotor forces below are expressed in the **body frame**;
position, velocity, gravity, and the total force equation are expressed in the **world frame**. The rotation matrix
$R$ converts body-frame vectors into world-frame vectors.

#### 1.1 Coordinate frames and rotor geometry

The body origin is taken at the drone's centre of mass. The body axes are $x_b$ (forward), $y_b$ (sideways), and $z_b$
(upward). Rotor $i$ is located at

$$
\boxed{\mathbf r_i=\begin{bmatrix}x_i\\y_i\\0\end{bmatrix}}
$$

Here, $x_i$ and $y_i$ are that rotor's signed offsets from the centre of mass, in metres. The third coordinate is zero
because this simplified geometry places all four rotor centres in the same plane as the centre of mass. For a symmetric
quadrotor, the offsets are combinations of $+d$ and $-d$; the exact sign for each motor depends on the model's motor
numbering and coordinate convention. Do not assume a particular motor-number diagram unless it matches the code.

<p align="center"><img src="docs/physics_figures/motor_forces.png" alt="Quadrotor with four rotor thrust vectors and motor directions" width="760"></p>

*Figure 1. Four thrust-producing rotor units. The exact labels and spin directions must agree with the motor numbering in the code.*

The thrust produced by rotor $i$ is modelled as

$$
\boxed{\mathbf F_i=\begin{bmatrix}0\\0\\f_i\end{bmatrix}}
$$

where $f_i\geq 0$ is the rotor's thrust in newtons. The zero $x$- and $y$-components mean the thrust is assumed to act
along the body vertical axis; this is the usual fixed-pitch, fixed-axis quadrotor approximation. The total thrust is

$$
T=\sum_{i=1}^{4}f_i.
$$

#### 1.2 Deriving roll and pitch torque with the cross product

Torque tells us how strongly a force tends to rotate the drone around its centre of mass. For rotor $i$,

$$
\boxed{\boldsymbol\tau_i=\mathbf r_i\times\mathbf F_i}
$$

Substitute the actual position and thrust vectors—not just their magnitudes:

$$
\boldsymbol\tau_i=
\begin{bmatrix}x_i\\y_i\\0\end{bmatrix}
\times
\begin{bmatrix}0\\0\\f_i\end{bmatrix}
=
\begin{vmatrix}
\mathbf e_x&\mathbf e_y&\mathbf e_z\\
x_i&y_i&0\\
0&0&f_i
\end{vmatrix}.
$$

<p align="center"><img src="docs/physics_figures/torque_from_thrust.png" alt="Rotor lever arm and thrust force producing torque by cross product" width="760"></p>

*Figure 2. Lever arm and thrust force: torque direction follows the right-hand rule.*

Expanding the determinant gives

$$
\boxed{\boldsymbol\tau_i=
\begin{bmatrix}y_i f_i\\-x_i f_i\\0\end{bmatrix}}.
$$

So, under this coordinate convention, the torque from thrust at one rotor is

$$
\tau_{x,i}=y_i f_i,\qquad \tau_{y,i}=-x_i f_i,\qquad \tau_{z,i}=0.
$$

Adding the contributions from all four rotors gives the total roll and pitch torques:

$$
\boxed{\tau_x=\sum_{i=1}^{4}y_i f_i},\qquad
\boxed{\tau_y=-\sum_{i=1}^{4}x_i f_i}.
$$

**Why is the thrust-generated yaw torque zero in this cross product?** Both the lever arm and the vertical thrust have
no component that creates a moment about the vertical axis. Yaw is modelled separately using the reaction torque from
spinning rotors:

$$
\boxed{\tau_z=k_M\sum_{i=1}^{4}s_i f_i}.
$$

Here $k_M$ is the yaw-torque coefficient (units depend on the thrust units and model definition), and $s_i=+1$ or $-1$
encodes the spin direction using the sign convention in the implementation. Oppositely spinning rotors have opposite yaw
reaction torques. Equal opposing contributions cancel in a balanced hover.

#### 1.3 Motor mixer and hover

The mixer maps the four individual thrusts to the total thrust and three body torques:

$$
\begin{bmatrix}T\\\tau_x\\\tau_y\\\tau_z\end{bmatrix}
=
\begin{bmatrix}
1&1&1&1\\
y_1&y_2&y_3&y_4\\
-x_1&-x_2&-x_3&-x_4\\
k_Ms_1&k_Ms_2&k_Ms_3&k_Ms_4
\end{bmatrix}
\begin{bmatrix}f_1\\f_2\\f_3\\f_4\end{bmatrix}.
$$

This is the **motor mixer**. Given a desired collective thrust and desired torques, the controller/mixer solves for the
individual motor thrusts, subject to the physical minimum and maximum thrust limits. The inverse exists only if the
mixer matrix has full rank; in practice, commands also have to respect motor saturation.

In level hover, the thrust balances weight and the net roll, pitch, and yaw torques are zero:

$$
T=mg,\qquad f_1=f_2=f_3=f_4=\frac{mg}{4}.
$$

For the example mass $m=0.5\,\mathrm{kg}$ and $g=9.81\,\mathrm{m/s^2}$, each motor supplies
$f_i=0.5\times9.81/4\approx1.226\,\mathrm N$. This numerical value is valid only if the mass used by the code is
$0.5\,\mathrm{kg}$.

#### 1.4 Motor lag: why thrust does not change instantly

The controller asks for a thrust, but the motor/propeller response takes time. A common first-order model is

$$
\boxed{\dot f_i=\frac{f_{i,\mathrm{cmd}}-f_i}{\tau_m}}
$$

where $f_{i,\mathrm{cmd}}$ is the commanded thrust, $f_i$ is the current thrust, and $\tau_m$ is the motor time
constant. The documented value is $\tau_m=0.03\,\mathrm s$. For a constant command, its continuous-time response is

$$
f_i(t)=f_{i,\mathrm{cmd}}+\big(f_i(0)-f_{i,\mathrm{cmd}}\big)e^{-t/\tau_m}.
$$

After one time constant, about 63.2% of the initial thrust error has been removed. This lag matters because the drone
cannot produce the requested correction instantaneously. The exact discrete update depends on the integration method in
the code.

#### 1.5 From four rotor thrusts to the total thrust force

Keep the individual rotor thrusts and the total thrust distinct:

$$
\boxed{T=f_1+f_2+f_3+f_4=\sum_{i=1}^{4}f_i}
$$

Each $f_i$ is the upward force generated by one propeller/rotor, measured in newtons. $T$ is the combined thrust magnitude from all four rotors, also in newtons. Because the four thrust vectors are assumed parallel to the body $z_b$ axis, their sum in body coordinates is

$$
\boxed{\mathbf F_{\mathrm{thrust},b}=\sum_{i=1}^{4}\mathbf F_i=\begin{bmatrix}0\\0\\T\end{bmatrix}}
$$

This is the **collective thrust**. It mainly controls vertical force when the drone is level. The differences between individual thrusts also create roll and pitch torques through the lever arms derived above. Therefore, the same four motor outputs determine both the total force used in translation and the torques used in rotation; these are connected parts of one rigid-body model, not separate unrelated formulas.

For level hover, $T=mg$. For a symmetric quadrotor with equal loading, this gives $f_i=mg/4$. When the drone tilts, the thrust still points along body $z_b$, but its world-frame direction changes with the rotation matrix $R$.

#### 1.6 Translational dynamics: how forces move the drone

Newton's second law is $\sum\mathbf F=m\mathbf a$. The drone's translational velocity $\mathbf v$ and acceleration
$\dot{\mathbf v}$ in the **world frame** are related by

$$
\boxed{m\dot{\mathbf v}=\mathbf F_{\mathrm{thrust}}+\mathbf F_{\mathrm{gravity}}
+\mathbf F_{\mathrm{aero}}}
$$

The body-frame thrust vector is $[0,0,T]^\top$. To add it to world-frame forces, rotate it using the body-to-world
rotation matrix $R$:

$$
\mathbf F_{\mathrm{thrust}}=R\begin{bmatrix}0\\0\\T\end{bmatrix},\qquad
\mathbf F_{\mathrm{gravity}}=\begin{bmatrix}0\\0\\-mg\end{bmatrix}.
$$

With the baseline quadratic-drag term and the wind-force correction used in this README, the model is written as

$$
\boxed{m\dot{\mathbf v}=R\begin{bmatrix}0\\0\\T\end{bmatrix}
-mg\begin{bmatrix}0\\0\\1\end{bmatrix}
-c_d\lVert\mathbf v\rVert\mathbf v+\mathbf F_{\mathrm{wind}}.}
$$

Symbols: $m$ is mass (kg), $R$ is the rotation matrix, $T$ is total thrust (N), $g$ is gravitational acceleration,
$c_d$ is the translational drag coefficient for this model, and $\mathbf F_{\mathrm{wind}}$ is the model's wind-force
correction. All terms on the right are forces in newtons. Divide the net force by $m$ to obtain acceleration.

<p align="center"><img src="docs/physics_figures/wind_force_balance.png" alt="Drone experiencing thrust, gravity, aerodynamic drag, and wind" width="760"></p>

*Figure 3. Force balance during flight with wind. Arrows are explanatory; their sizes depend on the simulated state.*

**What the equation means in different cases:**

- **Level hover:** $R[0,0,T]^\top=[0,0,mg]^\top$ when $T=mg$. Thrust cancels gravity, so vertical acceleration is zero if the other forces are zero.
- **Climb:** thrust has a world-vertical component larger than the weight, producing positive vertical acceleration (before drag and wind are considered).
- **Tilted flight:** thrust still points along body $z_b$, but $R$ rotates it so that part of the thrust acts horizontally. That horizontal component accelerates the drone sideways.
- **Moving through still air:** drag opposes the velocity under the stated quadratic-drag approximation. Its magnitude grows with speed squared because $\lVert\mathbf v\mathbf\rVert\mathbf v$ has magnitude $\lVert\mathbf v\rVert^2$.
- **Wind:** the aerodynamic force changes with air velocity relative to the drone. The wind model is described in Step 5; do not interpret the wind as an extra thrust command.

The baseline-drag term and the wind-force correction must be defined consistently in the implementation. If $\mathbf F_{\mathrm{wind}}$
is itself defined as the *entire* change from still-air drag to relative-air drag, do not accidentally count baseline drag twice.

#### 1.7 Rotational dynamics: why $\boldsymbol\tau=I\boldsymbol\alpha$ is not the full 3-D equation

For a single fixed axis, or when rotational coupling is negligible, the familiar relation is

$$
\tau=I\alpha.
$$

It says torque equals moment of inertia multiplied by angular acceleration. This is a useful one-axis approximation, but
a quadrotor is a **three-dimensional rigid body**. Its angular velocity changes direction as well as magnitude, and the
body axes rotate with the vehicle. In body coordinates, the full rigid-body Euler equation is

$$
\boxed{\boldsymbol\tau=I\dot{\boldsymbol\omega}
+\boldsymbol\omega\times(I\boldsymbol\omega)}
$$

or, equivalently,

$$
\boxed{I\dot{\boldsymbol\omega}=\boldsymbol\tau
-\boldsymbol\omega\times(I\boldsymbol\omega).}
$$

Here $\boldsymbol\omega=[p,q,r]^\top$ is the body angular velocity (roll, pitch, yaw rates), $I$ is the inertia tensor
about the centre of mass, and $\boldsymbol\tau=[\tau_x,\tau_y,\tau_z]^\top$ is the total applied body torque. The
cross-product term accounts for gyroscopic/inertial coupling between rotations about different axes. It cannot generally
be discarded during 3-D motion.

For a rigid body whose body axes align with its principal inertia axes, the inertia tensor is diagonal:

$$
I=\begin{bmatrix}I_{xx}&0&0\\0&I_{yy}&0\\0&0&I_{zz}\end{bmatrix}.
$$

Writing Euler's equation component by component gives

$$
\boxed{I_{xx}\dot p=\tau_x+(I_{yy}-I_{zz})qr}
$$
$$
\boxed{I_{yy}\dot q=\tau_y+(I_{zz}-I_{xx})rp}
$$
$$
\boxed{I_{zz}\dot r=\tau_z+(I_{xx}-I_{yy})pq.}
$$

These equations show why roll, pitch, and yaw are coupled in general: the acceleration around one axis can depend on
angular rates around the other two axes. If $p,q,r$ are all small, or if the relevant inertia differences/coupling terms
are negligible, the cross terms may be small and the approximation $\dot p\approx\tau_x/I_{xx}$ can be useful. It is
not a general replacement for the 3-D rigid-body equation.

<p align="center"><img src="docs/physics_figures/rotational_drag.png" alt="Drone rotating with opposing rotational drag torque" width="760"></p>

*Figure 4. Rotational drag opposes the current angular velocity; it is distinct from motor reaction torque.*

If the model includes rotational aerodynamic drag, its torque must be included in the **total** torque. For example, a
quadratic opposing-drag model can be written

$$
\boldsymbol\tau_{\mathrm{rot.drag}}=-c_{\mathrm{rot}}\lVert\boldsymbol\omega\rVert\boldsymbol\omega,
$$

where $c_{\mathrm{rot}}$ is the rotational-drag coefficient. It is zero at zero angular velocity and opposes the current
rotation. Then

$$
\boldsymbol\tau_{\mathrm{total}}=\boldsymbol\tau_{\mathrm{motors}}+
\boldsymbol\tau_{\mathrm{rot.drag}}+\boldsymbol\tau_{\mathrm{other}},
$$

where $\boldsymbol\tau_{\mathrm{other}}$ is included only if the code models other torques, such as a wind disturbance
torque. Do not include a term merely because it is physically possible; it must correspond to the implemented model.

#### 1.9 Why tilting creates sideways acceleration

In level hover, the thrust vector points vertically upward. When the drone pitches by $\theta$ or rolls by $\phi$, the
rotation matrix gives the thrust a horizontal component. Near level attitude, and using the sign convention in this model,

$$
\boxed{a_x\approx g\theta,\qquad a_y\approx-g\phi.}
$$

<p align="center"><img src="docs/physics_figures/tilt_sideways_motion.png" alt="Tilted quadrotor with thrust vector and horizontal acceleration" width="760"></p>

*Figure 5. Tilting redirects thrust and creates a horizontal force component.*

These are **small-angle approximations**: angles must be in radians, the drone is near hover, and the relation neglects
higher-order terms, drag, and wind. The sign follows the chosen body/world axes. At larger angles, use the full rotated
thrust vector $R[0,0,T]^\top$ rather than these approximations.

#### 1.10 How the equations connect in the simulation

<p align="center"><img src="docs/physics_figures/motor_lag.png" alt="Commanded motor thrust and lagged actual thrust response" width="760"></p>

*Figure 6. Actual thrust follows a step command gradually under a first-order motor-lag model.*

The full cause-and-effect sequence is:

1. The position/velocity and attitude controllers calculate desired thrust and torques.
2. The mixer converts $[T,\tau_x,\tau_y,\tau_z]^\top$ into four requested rotor thrusts.
3. Motor lag determines the thrust actually delivered by each rotor.
4. The four thrusts produce total thrust and roll/pitch torque through $\mathbf r_i\times\mathbf F_i$; rotor reaction torque contributes yaw torque.
5. Newton's translational equation calculates linear acceleration from rotated thrust, gravity, drag, and wind.
6. Euler's rigid-body equation calculates body angular acceleration from torque, inertia, and rotational coupling.
7. The integrator advances velocity, position, angular velocity, and attitude. The controller reads the updated state and corrects the next command.

<p align="center"><img src="docs/physics_figures/physics_pipeline.png" alt="Physics and control pipeline from commands to state feedback" width="900"></p>

*Figure 7. How motor commands, forces, rigid-body dynamics, integration, and feedback connect.*

Physics runs at 240 Hz and the controller at 120 Hz in the documented setup. Runge–Kutta 4 (RK4) is a numerical method
for integrating the differential equations; it does not change the physical laws themselves. The simulation's precise
state-update details depend on the implementation.

**Nonlinearity:** the rotation matrix contains sines and cosines; 3-D rotational dynamics contain products such as $pq$,
$pr$, and $qr$; quadratic drag depends nonlinearly on velocity; motor thrust has lag and may be saturated. The model can
be approximately linearised near hover, but the full model is nonlinear.

**Validation note:** report an RMSE value only with the corresponding test setup, variables, units, and comparison method.
A statement such as “RMSE about $10^{-3}$ or less” should be retained only if that value was measured and can be reproduced
from the project's validation script.

---

### Step 2 — Self-stabilization with a mathematical PID (`self_stabilizing_pid_ml.py`)

For any controlled quantity with reference $r$ and measurement $y$:

$$
e=r-y,\qquad u=\underbrace{K_p\,e}_{P}+\underbrace{K_i\!\int e\,dt}_{I}+\underbrace{K_d\,\dot e}_{D}
$$

The integral is clamped (anti-windup) and the derivative is the analytic rate error or a filtered difference. The class
logs $e$, $\int e$, $\dot e$, $P$, $I$, $D$ at every step, so each term is plotted individually.
**P** reacts to the present error, **I** removes steady offset (for example from wind), **D** damps oscillation.

**Cascaded control:**

1. position or velocity PID $\rightarrow$ desired acceleration $a_x,a_y$
2. acceleration $\rightarrow$ desired tilt: $\theta_{des}=\arctan(a_x/g),\ \phi_{des}=-\arctan(a_y/g)$ (limit 0.5 rad)
3. altitude PID $\rightarrow$ thrust: $T=\dfrac{m(g+a_z)}{\cos\phi\cos\theta}$
4. attitude PID $\rightarrow$ torque: $\tau=I\cdot\alpha_{PID}$ (roll, pitch, yaw)
5. mixer inverse $\rightarrow$ four motor thrusts

| Loop | $K_p$ | $K_i$ | $K_d$ |
|---|---|---|---|
| position (x, y) | 3.0 | 1.0 | 3.5 |
| altitude | 12 | 4 | 6 |
| attitude (roll, pitch) | 70 | 15 | 14 |

The fast inner (attitude) loop and slow outer (position) loop keep the cascade stable.

**Experiments**

* **Attitude disturbance:** roll and pitch torque kicks are rejected; $e$, $\int e$, $\dot e$, $P$, $I$, $D$ are plotted.
* **Circular trajectory with constant wind (0.6 N from t = 6 s):** the position loop is run as P, PI, PD and PID and
  the flown path is compared with the reference circle.

| Position loop | RMSE (m) | Final error (m) | Behaviour |
|---|---|---|---|
| P | 2.05 | 2.82 | undamped, unstable |
| PI | 3.96 | 5.16 | integral adds lag, unstable without D |
| PD | 0.36 | 0.46 | damped, but offset remains in wind |
| **PID** | **0.18** | **0.14** | damped and offset removed |

---

### Step 3 — Data-driven system identification (`self_stabilizing_pid_ml.py`)

Question: *given the controller input at one step, where will the drone be at the next step?* The state and input are

$$
\mathbf x=[\phi,\theta,p,q,r,v_x,v_y,v_z]^\top,\qquad \mathbf u=[T-mg,\ \tau_x,\ \tau_y,\ \tau_z]^\top
$$

The drone is flown with random targets and random motor dither, and triplets $(\mathbf x_k,\mathbf u_k,\mathbf x_{k+1})$
are recorded (90 s training, 40 s unseen test flight).

**Linear model (pseudo-inverse).** Assume $\mathbf x_{k+1}=A\mathbf x_k+B\mathbf u_k+\mathbf c$. Stacking all samples,
$X^{+}=\Theta Z$ with $Z=[X;\,U;\,\mathbf 1]$ and $\Theta=[A\ B\ \mathbf c]$, and the least-squares solution is

$$
\Theta=X^{+}\,\mathrm{pinv}(Z)
$$

**Nonlinear model (SINDy).** The derivative is written as a combination of candidate nonlinear functions:

$$
\dot{\mathbf x}=\Theta(\mathbf x,\mathbf u)\,\Xi
$$

The library contains the states and inputs plus $\sin\phi$, $\sin\theta\cos\phi$, $\cos\phi\cos\theta-1$, the products
$pq$, $pr$, $qr$, thrust-times-angle terms and quadratic drag $\lVert\mathbf v\rVert\mathbf v$. Coefficients $\Xi$ come
from sequentially thresholded least squares (pseudo-inverse, drop terms below 1 % of the target, refit), which leaves a
**sparse, interpretable** model. For example it recovers $\dot p\approx\tau_x/I_{xx}=133\,\tau_x$.

**Comparison on unseen data** (0.5 s open-loop prediction, recorded inputs, state never corrected):

| Model | roll $\phi$ (rad) | pitch $\theta$ (rad) | $v_x$ (m/s) |
|---|---|---|---|
| Linear $(A,B)$ | 0.0191 | 0.0175 | 0.1515 |
| **Nonlinear (SINDy)** | **0.0010** | **0.0005** | **0.0184** |
| Analytic physics | 0.0097 | 0.0102 | 0.0653 |

The drone is clearly nonlinear away from hover: the nonlinear identified model is about 10 times more accurate than the
linear one. (It also beats the analytic model, which assumes yaw = 0 and a simple integrator.)

---

### Step 4 — Intelligent obstacle avoidance (`obstacle_avoidance_pgmpy.py`)

**Features.** For an obstacle at $\mathbf p_o$ (radius $r_o$) and the drone at $\mathbf p$ (radius $r_d$):

$$
d=\lVert\mathbf p_o-\mathbf p\rVert-r_o-r_d,\qquad v_c=-(\mathbf v_o-\mathbf v)\cdot\mathbf n,\qquad \mathrm{TTC}=\frac{d}{v_c}\ \ (v_c>0.05)
$$

where $\mathbf n$ is the unit vector towards the obstacle and $v_c>0$ means approaching. Roll and pitch are added as
features because a strongly tilted drone brakes worse.

**Risk prediction.** 20 000 synthetic samples are labelled SAFE / WARNING / DANGER with a distance and TTC rule plus
5 % label noise (so accuracy measures how well the rule is learned, not real-world risk). Two models are trained:

* **Logistic Regression** on $[d,v_c,\mathrm{TTC},|\phi|,|\theta|]$.
* **pgmpy Bayesian network** on discretised features, with Distance $\rightarrow$ TTC $\leftarrow$ ClosingSpeed and
  Distance, TTC, Tilt $\rightarrow$ Risk. Its probability tables are learned from data and exact inference gives
  $P(\text{Risk}\mid\text{evidence})$.

| Model (held-out data) | Accuracy |
|---|---|
| Logistic Regression | 0.87 |
| pgmpy Bayesian network | 0.78 |

The network is less accurate (discretisation) but gives interpretable probabilities.

**Decision controller (minimum expected cost).**

$$
a^*=\arg\min_a\sum_r P(r)\,C[a][r],\qquad
C=\begin{bmatrix}0&3&10\\1&1&5\\3&2&1\end{bmatrix}
$$

(rows CONTINUE, SLOW_DOWN, AVOID; columns SAFE, WARNING, DANGER). Continuing in danger costs 10, avoiding when safe
only 3.

**Avoidance logic.** CONTINUE flies to the goal at 1.5 m/s; SLOW_DOWN cuts speed to 60 % and side-steps if the
obstacle is close and approaching; AVOID dodges sideways with a small retreat. The passing side is chosen once per
obstacle (to avoid chattering) by probing the free space on both sides. In the realistic world a small repulsion from
nearby obstacles and a low-pass filter on the velocity command are added.

**Result: obstacles moving at increasing speed** (30 random crossing scenarios per speed):

| Obstacle speed (m/s) | Collision rate, no avoidance | Collision rate, with avoidance |
|---|---|---|
| 0.5 | 100 % | 0 % |
| 1.0 | 70 % | 7 % |
| 2.0 | 67 % | 0 % |
| 3.0 | 53 % | 3 % |
| 4.0 | 53 % | 13 % |

Without avoidance the rate falls with speed partly because fast obstacles often cross before the drone arrives. The
3-obstacle zig-zag is passed below–above–below with no collision.

---

### Step 5 — Wind and the realistic world (`realistic_simulation.py`)

The wind is a mean wind plus Ornstein–Uhlenbeck turbulence plus random gusts with a "1−cos" shape, and the force uses
the air velocity **relative** to the drone:

$$
\mathbf w(t)=\mathbf w_{mean}\big(1+0.6\,g(t)\big)+\mathbf w_{turb}(t),\qquad
\mathbf F_{wind}=c_d\Big(\lVert\mathbf v\rVert\mathbf v-\lVert\mathbf v-\mathbf w\rVert(\mathbf v-\mathbf w)\Big)
$$

The world has a park path with 6 trees, 5 buildings with windows, a shed, a patrolling bird and a windsock. In the 3-D
window the drone shows a wind arrow and a label with wind speed, drone speed and the current decision, and the trees
sway with the wind.

---

### Step 6 — Integration and evaluation (`full_simulation.py`, `realistic_simulation.py`)

Every control step the system reads the state, computes obstacle features, predicts risk, decides, computes the
velocity command, runs the velocity and attitude PIDs, allocates motor thrusts and advances the physics. Metrics:
collision, goal reached, minimum clearance, time, maximum tilt, maximum altitude error.

| Test | Result |
|---|---|
| Zig-zag, 3 obstacles (PyBullet) | no collision, altitude error $\le$ 0.01 m |
| Crossing obstacles 0.5–3 m/s, with avoidance (6 runs each) | 0 % collisions at 0.5, 2, 3 m/s; 33 % at 1 m/s (small sample) |
| Realistic world, wind 0 / 2 / 4 m/s (3 runs each) | goal reached without collision, altitude within about 0.03 m |
| Realistic world, wind 6 m/s | collisions occurred; without avoidance every run collided |

---

## FILES AND HOW TO RUN

| File | Contents |
|---|---|
| `physics_model.py` | mixer, motor lag, Newton–Euler model, PyBullet drone, validation |
| `self_stabilizing_pid_ml.py` | PID, cascaded controller, experiments, system identification |
| `obstacle_avoidance_pgmpy.py` | features, Logistic Regression, pgmpy network, decision, avoidance, tests |
| `full_simulation.py` | complete system in PyBullet with metrics |
| `realistic_simulation.py` | realistic world with trees, buildings, wind |

```bash
pip install numpy pandas matplotlib scikit-learn pgmpy pybullet

python physics_model.py
python self_stabilizing_pid_ml.py
python obstacle_avoidance_pgmpy.py
python full_simulation.py          # 3-D PyBullet window; add --headless for plots and metrics only
python realistic_simulation.py     # 3-D window with trees, buildings and wind (--wind-speed 6 for stronger wind)
```

All figures are saved in the `plots/` folder.

---

## IMPLEMENTED, TESTED, AND STILL LIMITED

* **Implemented:** PID with separate P, I, D terms; physics with mixer and motor lag; linear and nonlinear system
  identification; Logistic Regression and Bayesian-network risk models; expected-cost decision; avoidance logic; full
  PyBullet integration; realistic world with wind.
* **Tested:** physics against the analytic model; disturbance rejection; P/PI/PD/PID comparison; identification on
  unseen data; classifier accuracy; obstacles at increasing speeds; wind sweep.
* **Limitations and next steps:** risk labels are rule-based and synthetic; the Bayesian network is less accurate than
  Logistic Regression; the planner is reactive (no global planning), so very fast obstacles and strong wind can still
  cause collisions; obstacles are 2-D circles at a fixed altitude; PyBullet sample sizes are small; the identified model
  is not yet used inside the controller (for example model-predictive control).

---

## REFERENCES

1. López-Sánchez, I., & Moreno-Valenzuela, J. (2023). PID Control of Quadrotor UAVs: A Survey. *Annual Reviews in Control*, 56, 100900.
2. López-Sánchez, I., & Moreno-Valenzuela, J. (2019). Position and Attitude Control of Multi-Rotor Aerial Vehicles: A Survey. *Annual Reviews in Control*.
3. Lotufo, M. A., et al. (2019). UAV Quadrotor Attitude Control: An ADRC-EMC Combined Approach. *Control Engineering Practice*, 84, 13–22.
4. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.
5. Ankan, A., & Panda, A. (2015). pgmpy: Probabilistic Graphical Models using Python. *Proceedings of the 14th Python in Science Conference*.
6. Coumans, E., & Bai, Y. PyBullet, a Python module for physics simulation for games, robotics and machine learning.

## LICENSE

MIT License.
