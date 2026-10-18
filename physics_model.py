"""
physics_model.py   (FILE 3 of 4)  -  The PHYSICS that lets the drone fly and stay stable
==========================================================================================
Run alone :  python physics_model.py          (validation + hover test + plots)
Imported by: self_stabilizing_pid_ml.py, full_simulation.py

WHAT IS IN THIS FILE
--------------------
1. DroneParams            : mass, arm length, inertia, thrust limits, drag, motor lag.
2. Mixer (X-configuration): converts  [T, tau_x, tau_y, tau_z]  <->  4 motor thrusts f1..f4
        T      =  f1 + f2 + f3 + f4                       (total thrust, N)
        tau_x  =  sum( y_i * f_i )                        (roll torque)
        tau_y  = -sum( x_i * f_i )                        (pitch torque)
        tau_z  =  kM * sum( s_i * f_i )                   (yaw torque, s_i = +-1 prop spin)
3. Newton-Euler equations (analytic NONLINEAR model of a quadrotor):
        phi_dot   = p + (q sin(phi) + r cos(phi)) tan(theta)       <- Euler kinematics (nonlinear)
        theta_dot = q cos(phi) - r sin(phi)
        p_dot     = ((Iyy-Izz) q r + tau_x) / Ixx                  <- gyroscopic coupling (nonlinear)
        q_dot     = ((Izz-Ixx) p r + tau_y) / Iyy
        r_dot     = ((Ixx-Iyy) p q + tau_z) / Izz
        v_dot     = (1/m) * R(phi,theta,psi) * [0,0,T] - [0,0,g] - drag(v)/m   <- thrust direction (nonlinear)
4. Quadrotor class        : the same physics inside PyBullet (rigid body + gravity + motor lag
                            + quadratic drag + external wind force / disturbance torque).

WHY THE DRONE IS A NONLINEAR SYSTEM  (say this in your review)
---------------------------------------------------------------
 * trigonometric terms (sin, cos, tan of roll/pitch)  -> thrust direction & Euler kinematics
 * products of states  (q*r, p*r, p*q)                -> gyroscopic coupling
 * quadratic aerodynamic drag  |v|v                   -> non-polynomial-degree-1 damping
 * actuator saturation + motor lag                    -> non-smooth / dynamic input
 Near hover (small angles) sin(x)~x, cos(x)~1 and products vanish, so the drone
 behaves almost like  x_{k+1} = A x_k + B u_k.  Far from hover it does not.
"""
import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data

G = 9.81
SIM_DT = 1.0 / 240.0          # PyBullet step
SUBSTEPS = 2                  # physics steps per control step
CTRL_DT = SIM_DT * SUBSTEPS   # control period = 1/120 s


class DroneParams:
    m = 0.5                    # kg
    L = 0.17                   # m, centre -> motor
    Ixx = 0.0075               # kg m^2
    Iyy = 0.0075
    Izz = 0.0135
    kM = 0.016                 # yaw torque per unit thrust (m)
    f_max = 3.0                # N per motor  (thrust/weight = 12/4.9 = 2.4)
    tau_motor = 0.03           # s, first-order motor lag
    c_drag = 0.05              # N/(m/s)^2  quadratic translational drag
    c_rot = 0.001              # N m/(rad/s)^2 quadratic rotational drag


PRM = DroneParams()

# ---------------------------------------------------------------- mixer (X configuration, front = +x)
_d = PRM.L / np.sqrt(2.0)
MOTOR_X = np.array([+_d, -_d, -_d, +_d])      # motor positions in body frame
MOTOR_Y = np.array([-_d, -_d, +_d, +_d])
MOTOR_S = np.array([-1.0, +1.0, -1.0, +1.0])  # propeller spin direction
MIXER = np.vstack([np.ones(4),                # T
                   MOTOR_Y,                   # tau_x
                   -MOTOR_X,                  # tau_y
                   PRM.kM * MOTOR_S])         # tau_z
MIXER_INV = np.linalg.inv(MIXER)
HOVER_F = PRM.m * G / 4.0                     # thrust per motor to hover


def wrench_to_motors(T, tx, ty, tz):
    """[T,tau] -> 4 motor thrusts (clipped to physical limits)."""
    f = MIXER_INV @ np.array([T, tx, ty, tz])
    return np.clip(f, 0.0, PRM.f_max)


def motors_to_wrench(f):
    return MIXER @ np.asarray(f)


def rot_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


# ---------------------------------------------------------------- analytic Newton-Euler model
# reduced state  x = [phi, theta, p, q, r, vx, vy, vz]  (yaw held ~0),  input u = [T, tau_x, tau_y, tau_z]
STATE_NAMES = ["phi", "theta", "p", "q", "r", "vx", "vy", "vz"]
INPUT_NAMES = ["T", "tau_x", "tau_y", "tau_z"]


def physics_derivative(x, u, yaw=0.0):
    """Continuous-time analytic model  x_dot = f(x,u)  (the 'true physics')."""
    phi, theta, pp, q, r, vx, vy, vz = x
    T, tx, ty, tz = u
    P = PRM
    phi_d = pp + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_d = q * np.cos(phi) - r * np.sin(phi)
    w = np.array([pp, q, r])
    rot_drag = P.c_rot * np.linalg.norm(w) * w
    p_d = ((P.Iyy - P.Izz) * q * r + tx - rot_drag[0]) / P.Ixx
    q_d = ((P.Izz - P.Ixx) * pp * r + ty - rot_drag[1]) / P.Iyy
    r_d = ((P.Ixx - P.Iyy) * pp * q + tz - rot_drag[2]) / P.Izz
    v = np.array([vx, vy, vz])
    R = rot_matrix(phi, theta, yaw)
    acc = (R @ np.array([0, 0, T])) / P.m - np.array([0, 0, G]) - P.c_drag * np.linalg.norm(v) * v / P.m
    return np.array([phi_d, theta_d, p_d, q_d, r_d, acc[0], acc[1], acc[2]])


# ---------------------------------------------------------------- PyBullet drone
class Quadrotor:
    def __init__(self, gui=False, start_pos=(0.0, 0.0, 1.0), fancy=None, realtime=True):
        self.gui = gui
        self.realtime = realtime and gui
        fancy = gui if fancy is None else fancy
        if gui:
            opts = "--opengl2" if ("--opengl2" in sys.argv or os.environ.get("DRONE_OPENGL2") == "1") else ""
            self.cid = p.connect(p.GUI, options=opts)
            for flag in (p.COV_ENABLE_GUI, p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.cid)      # hide side panels (faster, cleaner)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -G, physicsClientId=self.cid)
        p.setTimeStep(SIM_DT, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        c = self.cid
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[PRM.L, PRM.L, 0.02], physicsClientId=c)
        if not fancy:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[PRM.L, 0.015, 0.015],
                                      rgbaColor=[0.1, 0.1, 0.1, 1], physicsClientId=c)
            kw = {}
        else:   # nicer quadcopter look: centre body + 2 crossed arms + 4 coloured motor discs (visual only, mass 0)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.025], rgbaColor=[0.15, 0.15, 0.2, 1],
                                      physicsClientId=c)
            arm = [p.createVisualShape(p.GEOM_BOX, halfExtents=[PRM.L * 1.0, 0.012, 0.008],
                                       rgbaColor=[0.05, 0.05, 0.05, 1], physicsClientId=c) for _ in range(2)]
            disc_front = p.createVisualShape(p.GEOM_CYLINDER, radius=0.07, length=0.008,
                                             rgbaColor=[0.9, 0.1, 0.1, 0.9], physicsClientId=c)
            disc_back = p.createVisualShape(p.GEOM_CYLINDER, radius=0.07, length=0.008,
                                            rgbaColor=[0.1, 0.4, 0.9, 0.9], physicsClientId=c)
            ids = [arm[0], arm[1]] + [disc_front if MOTOR_X[i] > 0 else disc_back for i in range(4)]
            pos_l = [[0, 0, 0.0], [0, 0, 0.0]] + [[MOTOR_X[i], MOTOR_Y[i], 0.018] for i in range(4)]
            q45 = p.getQuaternionFromEuler([0, 0, np.pi / 4])
            qm45 = p.getQuaternionFromEuler([0, 0, -np.pi / 4])
            orn_l = [q45, qm45] + [[0, 0, 0, 1]] * 4
            n = len(ids)
            kw = dict(linkMasses=[0] * n, linkCollisionShapeIndices=[-1] * n, linkVisualShapeIndices=ids,
                      linkPositions=pos_l, linkOrientations=orn_l,
                      linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
                      linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n)
        self.body = p.createMultiBody(baseMass=PRM.m, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                      basePosition=list(start_pos), physicsClientId=c, **kw)
        p.changeDynamics(self.body, -1, localInertiaDiagonal=[PRM.Ixx, PRM.Iyy, PRM.Izz],
                         linearDamping=0.0, angularDamping=0.0, physicsClientId=c)
        self.f = np.full(4, HOVER_F)          # actual motor thrusts (after lag)
        self._last_pos = np.array(start_pos, float)
        self._wall = time.time()
        self._cam_n = 0
        if gui:
            p.resetDebugVisualizerCamera(3.5, 50, -30, list(start_pos), physicsClientId=c)

    # -------------------------------------------------------
    def reset(self, pos, yaw=0.0):
        p.resetBasePositionAndOrientation(self.body, list(pos), p.getQuaternionFromEuler([0, 0, yaw]),
                                          physicsClientId=self.cid)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        self.f = np.full(4, HOVER_F)
        self._last_pos = np.array(pos, float)

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    # -------------------------------------------------------
    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        vel, w_world = p.getBaseVelocity(self.body, physicsClientId=self.cid)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        w_body = R.T @ np.array(w_world)
        return dict(pos=np.array(pos), vel=np.array(vel), roll=roll, pitch=pitch, yaw=yaw,
                    rates=w_body, f=self.f.copy())

    # -------------------------------------------------------
    def step(self, f_cmd, wind_force=(0, 0, 0), dist_torque=(0, 0, 0)):
        """Advance one CONTROL period (SUBSTEPS physics steps).
        f_cmd : commanded motor thrusts (4,)
        returns the average wrench u=[T,tau_x,tau_y,tau_z] that was really applied."""
        f_cmd = np.clip(np.asarray(f_cmd, float), 0.0, PRM.f_max)
        wind_force = np.asarray(wind_force, float)
        dist_torque = np.asarray(dist_torque, float)
        u_sum = np.zeros(4)
        alpha = SIM_DT / (PRM.tau_motor + SIM_DT)             # discrete 1st-order motor lag
        for _ in range(SUBSTEPS):
            self.f += alpha * (f_cmd - self.f)
            u = MIXER @ self.f
            u_sum += u
            pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
            vel, w_world = p.getBaseVelocity(self.body, physicsClientId=self.cid)
            R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            v = np.array(vel)
            w_body = R.T @ np.array(w_world)
            F_world = R @ np.array([0, 0, u[0]]) - PRM.c_drag * np.linalg.norm(v) * v + wind_force
            tau_body = u[1:] - PRM.c_rot * np.linalg.norm(w_body) * w_body + R.T @ dist_torque
            p.applyExternalForce(self.body, -1, F_world.tolist(), pos, p.WORLD_FRAME, physicsClientId=self.cid)
            p.applyExternalTorque(self.body, -1, (R @ tau_body).tolist(), p.WORLD_FRAME, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)
        if self.gui:       # draw the flown path in the GUI
            new = np.array(p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)[0])
            p.addUserDebugLine(self._last_pos.tolist(), new.tolist(), [1, 0.5, 0], 3, 0, physicsClientId=self.cid)
            self._last_pos = new
            self._cam_n += 1
            if self._cam_n % 6 == 0:          # camera follows the drone
                cam = p.getDebugVisualizerCamera(physicsClientId=self.cid)
                p.resetDebugVisualizerCamera(cam[10], cam[8], cam[9], new.tolist(), physicsClientId=self.cid)
            if self.realtime:                  # slow down to real time so you can watch
                self._wall = max(self._wall, time.time() - 0.05) + CTRL_DT
                dt_sleep = self._wall - time.time()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
        return u_sum / SUBSTEPS


# ---------------------------------------------------------------- validation: PyBullet vs analytic physics
def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def validate_against_analytic(duration=0.4):
    """Constant, motor-lag-free-ish input: small roll+pitch torque. Compare PyBullet to Newton-Euler."""
    drone = Quadrotor(gui=False, start_pos=(0, 0, 5.0))
    f_cmd = HOVER_F + np.array([-0.02, -0.02, +0.02, +0.02]) * 2 + np.array([-0.01, +0.01, +0.01, -0.01])
    x = np.zeros(8)
    n = int(duration / CTRL_DT)
    errs, hist_b, hist_a, ts = [], [], [], []
    for k in range(n):
        u = drone.step(f_cmd)
        s = drone.get_state()
        # analytic model driven by the SAME applied wrench u (so motor lag is not counted as error)
        x = rk4(physics_derivative, x, u, CTRL_DT)
        xb = np.array([s["roll"], s["pitch"], *s["rates"], *s["vel"]])
        hist_b.append(xb); hist_a.append(x.copy()); ts.append((k + 1) * CTRL_DT)
    drone.close()
    hist_b, hist_a = np.array(hist_b), np.array(hist_a)
    return np.array(ts), hist_b, hist_a


def hover_test(duration=4.0):
    """Open loop (no controller): constant hover thrust. Shows why feedback is needed."""
    drone = Quadrotor(gui=False, start_pos=(0, 0, 3.0))
    ts, roll, z = [], [], []
    for k in range(int(duration / CTRL_DT)):
        t = k * CTRL_DT
        tq = (0.02, 0, 0) if 0.5 < t < 0.6 else (0, 0, 0)     # tiny kick at t=0.5 s
        drone.step(np.full(4, HOVER_F), dist_torque=tq)
        s = drone.get_state()
        ts.append(t); roll.append(np.degrees(s["roll"])); z.append(s["pos"][2])
    drone.close()
    return np.array(ts), np.array(roll), np.array(z)


if __name__ == "__main__":
    import matplotlib
    if "--no-show" in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("plots", exist_ok=True)
    np.set_printoptions(precision=4, suppress=True)
    print("Hover thrust per motor : %.3f N  (m*g/4)" % HOVER_F)
    print("Mixer matrix [T,tx,ty,tz] = MIXER @ [f1..f4]\n", MIXER)
    print("cond(MIXER) = %.2f  (invertible -> any wrench can be allocated)" % np.linalg.cond(MIXER))

    ts, xb, xa = validate_against_analytic()
    rmse = np.sqrt(np.mean((xb - xa) ** 2, axis=0))
    print("\nPyBullet vs analytic Newton-Euler, RMSE per state over 0.4 s:")
    for n_, r_ in zip(STATE_NAMES, rmse):
        print("   %-6s %.5f" % (n_, r_))

    t2, roll, z = hover_test()
    print("\nOpen-loop hover: roll after kick = %.1f deg at t=4 s  -> never recovers without feedback" % roll[-1])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, nm in zip([0, 1, 5], ["roll (rad)", "pitch (rad)", "vx (m/s)"]):
        ax[[0, 1, 5].index(i)].plot(ts, xb[:, i], label="PyBullet")
        ax[[0, 1, 5].index(i)].plot(ts, xa[:, i], "--", label="analytic Newton-Euler")
        ax[[0, 1, 5].index(i)].set_title(nm); ax[[0, 1, 5].index(i)].set_xlabel("t (s)"); ax[[0, 1, 5].index(i)].legend()
    plt.tight_layout(); plt.savefig("plots/physics_validation.png", dpi=120)
    plt.figure(figsize=(6, 3.5)); plt.plot(t2, roll); plt.xlabel("t (s)"); plt.ylabel("roll (deg)")
    plt.title("Open loop: a small kick is never corrected"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("plots/open_loop_hover.png", dpi=120)
    plt.show()