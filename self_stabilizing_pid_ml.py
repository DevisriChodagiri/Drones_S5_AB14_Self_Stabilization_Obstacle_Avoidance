"""
self_stabilizing_pid_ml.py   (FILE 2 of 4)  -  Self-stabilization + data-driven ML part
=======================================================================================
Run :  python self_stabilizing_pid_ml.py                 (everything, ~1-2 min)
       python self_stabilizing_pid_ml.py --exp pid       (PID experiments only)
       python self_stabilizing_pid_ml.py --exp sysid     (system identification only)
       add  --no-show  to only save the pictures into ./plots,  --gui for PyBullet window

PART A  -  MATHEMATICAL PID (every term separately)
    e(t)  = reference - measurement
    P     = Kp * e
    I     = Ki * integral_0^t e dt          (anti-windup clamp)
    D     = Kd * de/dt                       (de/dt = ref_rate - measured_rate, low-pass filtered)
    u     = P + I + D
  Cascaded control:
    position error --PID--> desired acceleration --> desired roll/pitch
    roll/pitch error --PID--> angular acceleration * inertia = torque
    altitude error --PID--> thrust          yaw error --PID--> yaw torque
    [T, tau_x, tau_y, tau_z] --mixer--> 4 motor thrusts      (physics_model.py)
  Experiments:
    1) attitude disturbance (torque kick)  -> roll & pitch response, e, integral(e), de/dt, P, I, D plotted
    2) circular trajectory + constant wind -> P-only, PD, PI, PID compared on the SAME path

PART B  -  DATA-DRIVEN SYSTEM IDENTIFICATION (the ML part)
    Collect data  (x_k, u_k) -> x_{k+1}   from the flying drone, then learn the dynamics:
    1) LINEAR      x_{k+1} = A x_k + B u_k + c        solved with the PSEUDO-INVERSE
                   Theta = X_next * pinv([x; u; 1])
    2) NONLINEAR   x_dot = Theta(x,u) * Xi            (SINDy: library of candidate nonlinear
                   functions: sin, cos, products q*r, drag |v|v, T*sin(theta) ... and sparse
                   regression - sequentially thresholded least squares using pinv)
    3) PHYSICS     analytic Newton-Euler model (physics_model.py) as reference
    Compared by one-step error and 0.5 s open-loop prediction error on unseen data.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_model import (Quadrotor, PRM, G, CTRL_DT, MIXER_INV, MIXER, HOVER_F,
                           STATE_NAMES, INPUT_NAMES, physics_derivative)

os.makedirs("plots", exist_ok=True)
np.set_printoptions(precision=4, suppress=True, linewidth=150)


# =====================================================================================
#                              PART A : MATHEMATICAL PID
# =====================================================================================
class PID:
    """u = Kp*e + Ki*int(e) + Kd*de/dt  with separate logging of e, int(e), de/dt, P, I, D."""

    def __init__(self, kp, ki, kd, dt=CTRL_DT, i_limit=5.0, d_filter=0.15, name=""):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.i_limit, self.d_filter, self.name = i_limit, d_filter, name
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.e_prev = None
        self.d_state = 0.0
        self.log = {k: [] for k in ["e", "int_e", "de_dt", "P", "I", "D", "u"]}

    def update(self, ref, meas, de_dt=None):
        e = ref - meas
        # integral of error (rectangular rule) with clamping so that |Ki*int| <= i_limit
        self.integral += e * self.dt
        if self.ki > 0:
            lim = self.i_limit / self.ki
            self.integral = float(np.clip(self.integral, -lim, lim))
        # derivative of error: given analytically (ref_rate - measured_rate) or numerically
        if de_dt is None:
            raw = 0.0 if self.e_prev is None else (e - self.e_prev) / self.dt
            self.d_state += self.d_filter * (raw - self.d_state)
            de_dt = self.d_state
        self.e_prev = e
        P, I, D = self.kp * e, self.ki * self.integral, self.kd * de_dt
        u = P + I + D
        for k, v in zip(self.log.keys(), [e, self.integral, de_dt, P, I, D, u]):
            self.log[k].append(float(v))
        return u


# gains  (kp, ki, kd)
DEFAULT_GAINS = dict(
    pos_xy=(3.0, 1.0, 3.5),      # position error [m]  -> acceleration [m/s^2]
    vel_xy=(4.0, 1.0, 0.05),     # velocity error [m/s]-> acceleration (used with avoidance)
    z=(12.0, 4.0, 6.0),          # altitude error      -> vertical acceleration
    att=(70.0, 15.0, 14.0),      # roll/pitch error [rad] -> angular acceleration [rad/s^2]
    yaw=(20.0, 2.0, 8.0),
)


class DroneController:
    """Cascaded PID.  Two modes:  position tracking  or  velocity tracking (for obstacle avoidance)."""

    TILT_MAX = 0.5   # rad

    def __init__(self, gains=None, pos_variant="PID"):
        g = dict(DEFAULT_GAINS)
        if gains:
            g.update(gains)
        kp, ki, kd = g["pos_xy"]
        mask = dict(P=(1, 0, 0), PI=(1, 1, 0), PD=(1, 0, 1), PID=(1, 1, 1))[pos_variant]
        kp, ki, kd = kp * mask[0], ki * mask[1], kd * mask[2]
        self.pid_x = PID(kp, ki, kd, name="x")
        self.pid_y = PID(kp, ki, kd, name="y")
        self.pid_vx = PID(*g["vel_xy"], name="vx")
        self.pid_vy = PID(*g["vel_xy"], name="vy")
        self.pid_z = PID(*g["z"], name="z")
        self.pid_roll = PID(*g["att"], name="roll")
        self.pid_pitch = PID(*g["att"], name="pitch")
        self.pid_yaw = PID(*g["yaw"], name="yaw")

    def reset(self):
        for v in vars(self).values():
            if isinstance(v, PID):
                v.reset()

    def update(self, s, z_ref=1.0, ref_pos=None, ref_vel=None, v_cmd=None, yaw_ref=0.0):
        pos, vel = s["pos"], s["vel"]
        roll, pitch, yaw = s["roll"], s["pitch"], s["yaw"]
        pr, qr, rr = s["rates"]
        # ---- outer loop : desired horizontal acceleration
        if v_cmd is not None:                                   # velocity mode
            ax = self.pid_vx.update(v_cmd[0], vel[0])
            ay = self.pid_vy.update(v_cmd[1], vel[1])
        else:                                                   # position mode
            rv = np.zeros(3) if ref_vel is None else ref_vel
            ax = self.pid_x.update(ref_pos[0], pos[0], de_dt=rv[0] - vel[0])
            ay = self.pid_y.update(ref_pos[1], pos[1], de_dt=rv[1] - vel[1])
            z_ref = ref_pos[2]
        vz_ref = 0.0 if (ref_vel is None or v_cmd is not None) else ref_vel[2]
        # ---- acceleration -> desired tilt (rotate into yaw frame; a_x = g*tan(pitch), a_y = -g*tan(roll))
        abx = np.cos(yaw) * ax + np.sin(yaw) * ay
        aby = -np.sin(yaw) * ax + np.cos(yaw) * ay
        pitch_des = float(np.clip(np.arctan(abx / G), -self.TILT_MAX, self.TILT_MAX))
        roll_des = float(np.clip(-np.arctan(aby / G), -self.TILT_MAX, self.TILT_MAX))
        # ---- altitude PID -> total thrust (tilt compensated)
        az = self.pid_z.update(z_ref, pos[2], de_dt=vz_ref - vel[2])
        T = PRM.m * (G + az) / max(np.cos(roll) * np.cos(pitch), 0.5)
        T = float(np.clip(T, 0.0, 4 * PRM.f_max))
        # ---- attitude PID -> torques (alpha = PID output, tau = I*alpha); reference rate = 0
        tx = PRM.Ixx * self.pid_roll.update(roll_des, roll, de_dt=-pr)
        ty = PRM.Iyy * self.pid_pitch.update(pitch_des, pitch, de_dt=-qr)
        e_yaw = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
        tz = PRM.Izz * self.pid_yaw.update(yaw + e_yaw, yaw, de_dt=-rr)
        f = np.clip(MIXER_INV @ np.array([T, tx, ty, tz]), 0.0, PRM.f_max)
        return f, dict(roll_des=roll_des, pitch_des=pitch_des, T=T)


# =====================================================================================
#                              experiment helpers
# =====================================================================================
def run_sim(ctrl, duration, ref_fn, dist_fn=None, gui=False, start=None, drone=None):
    """ref_fn(t)->(pos_ref, vel_ref);  dist_fn(t)->(wind_force, torque)."""
    own = drone is None
    pos0, _ = ref_fn(0.0)
    drone = drone or Quadrotor(gui=gui, start_pos=pos0 if start is None else start)
    drone.reset(pos0 if start is None else start)
    ctrl.reset()
    L = {k: [] for k in ["t", "pos", "ref", "roll", "pitch", "yaw", "roll_des", "pitch_des", "wind"]}
    for k in range(int(duration / CTRL_DT)):
        t = k * CTRL_DT
        s = drone.get_state()
        rp, rv = ref_fn(t)
        f, info = ctrl.update(s, ref_pos=rp, ref_vel=rv)
        wind, tq = dist_fn(t) if dist_fn else ((0, 0, 0), (0, 0, 0))
        drone.step(f, wind_force=wind, dist_torque=tq)
        for key, val in [("t", t), ("pos", s["pos"]), ("ref", rp), ("roll", s["roll"]), ("pitch", s["pitch"]),
                         ("yaw", s["yaw"]), ("roll_des", info["roll_des"]), ("pitch_des", info["pitch_des"]),
                         ("wind", np.array(wind))]:
            L[key].append(val)
    if own:
        drone.close()
    return {k: np.array(v) for k, v in L.items()}


def hover_ref(t):
    return np.array([0.0, 0.0, 1.0]), np.zeros(3)


def circle_ref(t, R=1.0, period=10.0, z=1.0):
    w = 2 * np.pi / period
    pos = np.array([R * np.cos(w * t), R * np.sin(w * t), z])
    vel = np.array([-R * w * np.sin(w * t), R * w * np.cos(w * t), 0.0])
    return pos, vel


def plot_pid_terms(pid, title, fname, t0=0.0):
    t = np.arange(len(pid.log["e"])) * CTRL_DT
    fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for a, (key, lab) in zip(ax.ravel(), [("e", "error e"), ("int_e", "integral of error  ∫e dt"),
                                          ("de_dt", "derivative  de/dt"), ("P", "P term  Kp·e"),
                                          ("I", "I term  Ki·∫e"), ("D", "D term  Kd·de/dt")]):
        a.plot(t, pid.log[key]); a.set_title(lab); a.grid(alpha=.3)
    for a in ax[1]:
        a.set_xlabel("t (s)")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(fname, dpi=120)


# =====================================================================================
#           EXPERIMENT 1 :  attitude disturbance  (torque kick)
# =====================================================================================
def experiment_attitude(gui=False):
    print("\n=== EXPERIMENT 1: attitude disturbance (roll kick at 2 s, pitch kick at 5 s) ===")

    def dist(t):
        if 2.0 <= t < 2.2:
            return (0, 0, 0), (0.05, 0, 0)
        if 5.0 <= t < 5.2:
            return (0, 0, 0), (0, -0.05, 0)
        return (0, 0, 0), (0, 0, 0)

    ctrl = DroneController()
    log = run_sim(ctrl, 8.0, hover_ref, dist, gui=gui)
    plt.figure(figsize=(9, 4))
    plt.plot(log["t"], np.degrees(log["roll"]), label="roll")
    plt.plot(log["t"], np.degrees(log["pitch"]), label="pitch")
    plt.axvspan(2.0, 2.2, color="r", alpha=.15); plt.axvspan(5.0, 5.2, color="r", alpha=.15)
    plt.xlabel("t (s)"); plt.ylabel("deg"); plt.title("Self-stabilization: roll & pitch after torque kicks")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("plots/exp1_roll_pitch.png", dpi=120)
    plot_pid_terms(ctrl.pid_roll, "ROLL PID : e, ∫e, de/dt, P, I, D (individually)", "plots/exp1_roll_pid_terms.png")
    plot_pid_terms(ctrl.pid_pitch, "PITCH PID : e, ∫e, de/dt, P, I, D (individually)", "plots/exp1_pitch_pid_terms.png")
    peak = np.degrees(np.max(np.abs(log["roll"][(log["t"] > 2) & (log["t"] < 4)])))
    settle = np.degrees(np.max(np.abs(log["roll"][(log["t"] > 4.5) & (log["t"] < 5)])))
    print("peak roll after kick : %.2f deg ,  roll 2.5 s after kick: %.3f deg" % (peak, settle))


# =====================================================================================
#           EXPERIMENT 2 :  circle trajectory + wind gust, P / PI / PD / PID
# =====================================================================================
def experiment_trajectory(gui=False):
    print("\n=== EXPERIMENT 2: circular trajectory with constant wind (Fx = 0.6 N from t = 6 s) ===")

    def wind(t):
        return ((0.6, 0.0, 0.0) if t >= 6.0 else (0, 0, 0)), (0, 0, 0)

    results, logs = {}, {}
    for var in ["P", "PI", "PD", "PID"]:
        ctrl = DroneController(pos_variant=var)
        lg = run_sim(ctrl, 24.0, circle_ref, wind, gui=gui and var == "PID")
        err = np.linalg.norm(lg["pos"] - lg["ref"], axis=1)
        results[var] = (np.sqrt(np.mean(err ** 2)), err.max(), err[-300:].mean())
        logs[var] = (lg, ctrl)
    print("%-5s %12s %12s %16s" % ("loop", "RMSE (m)", "max err (m)", "final err (m)"))
    for k, v in results.items():
        print("%-5s %12.4f %12.4f %16.4f" % (k, *v))

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for a, var in zip(ax, ["P", "PI", "PD", "PID"]):
        lg = logs[var][0]
        a.plot(lg["ref"][:, 0], lg["ref"][:, 1], "k--", label="reference path")
        a.plot(lg["pos"][:, 0], lg["pos"][:, 1], label="flown trajectory")
        a.set_title("%s controller   RMSE=%.3f m" % (var, results[var][0])); a.axis("equal"); a.grid(alpha=.3)
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig("plots/exp2_trajectories_xy.png", dpi=120)

    fig = plt.figure(figsize=(7, 6)); a3 = fig.add_subplot(111, projection="3d")
    lg = logs["PID"][0]
    a3.plot(*lg["ref"].T, "k--", label="reference"); a3.plot(*lg["pos"].T, label="PID flown")
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z"); a3.legend(); a3.set_title("3-D trajectory (PID)")
    plt.tight_layout(); plt.savefig("plots/exp2_trajectory_3d.png", dpi=120)

    plt.figure(figsize=(9, 4))
    for var in logs:
        lg = logs[var][0]
        plt.plot(lg["t"], np.linalg.norm(lg["pos"] - lg["ref"], axis=1), label=var)
    plt.axvspan(6, 24, color="c", alpha=.15, label="wind on")
    plt.xlabel("t (s)"); plt.ylabel("tracking error (m)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Effect of P, I, D on tracking error"); plt.tight_layout(); plt.savefig("plots/exp2_error.png", dpi=120)

    plot_pid_terms(logs["PID"][1].pid_x, "X-POSITION PID in the wind experiment: e, ∫e, de/dt, P, I, D",
                   "plots/exp2_x_pid_terms.png")


# =====================================================================================
#                      PART B : DATA-DRIVEN SYSTEM IDENTIFICATION
# =====================================================================================
def collect_data(duration=90.0, seed=0, gui=False):
    """Fly with random set-points + random motor dither (persistent excitation).
    Returns X (N+1,8) states, U (N,4) applied wrench deviations [T-mg, tau_x, tau_y, tau_z]."""
    rng = np.random.default_rng(seed)
    drone = Quadrotor(gui=gui, start_pos=(0, 0, 1.0))
    ctrl = DroneController()
    target = np.array([0.0, 0.0, 1.0]); X, U = [], []
    dither = np.zeros(4)
    for k in range(int(duration / CTRL_DT)):
        if k % 60 == 0:                         # new random target every 0.5 s
            target = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.7, 1.6)])
        if k % 6 == 0:
            dither = rng.normal(0, 0.12, 4)
        s = drone.get_state()
        X.append([s["roll"], s["pitch"], *s["rates"], *s["vel"]])
        f, _ = ctrl.update(s, ref_pos=target, ref_vel=np.zeros(3))
        u = drone.step(f + dither)
        u[0] -= PRM.m * G
        U.append(u)
        if np.linalg.norm(s["pos"][:2]) > 6 or abs(s["roll"]) > 1.0 or abs(s["pitch"]) > 1.0:
            drone.reset((0, 0, 1.0)); ctrl.reset()           # safety reset; the jump is removed below
            X.append(None); U.append(None)
    s = drone.get_state(); X.append([s["roll"], s["pitch"], *s["rates"], *s["vel"]])
    drone.close()
    # build (x_k, u_k, x_{k+1}) triplets and drop those that cross a reset
    xs, us, xn = [], [], []
    for i in range(len(U)):
        if U[i] is None or X[i] is None or X[i + 1] is None:
            continue
        xs.append(X[i]); us.append(U[i]); xn.append(X[i + 1])
    return np.array(xs), np.array(us), np.array(xn)


# ---------------- (1) linear model  x_{k+1} = A x + B u + c  via pseudo-inverse
def fit_linear(Xk, Uk, Xn):
    Z = np.vstack([Xk.T, Uk.T, np.ones(len(Xk))])               # (13, N)
    Theta = Xn.T @ np.linalg.pinv(Z)                            # (8, 13)   <-- pseudo-inverse
    A, B, c = Theta[:, :8], Theta[:, 8:12], Theta[:, 12]
    return A, B, c


# ---------------- (2) nonlinear library (SINDy)
def build_library(X, U):
    """Candidate functions Theta(x,u). X:(N,8) U:(N,4) (U[:,0] = T - m g).  Returns (N,K), names."""
    phi, th, p_, q, r, vx, vy, vz = X.T
    dT, tx, ty, tz = U.T
    sphi, sth, cphi, cth = np.sin(phi), np.sin(th), np.cos(phi), np.cos(th)
    cols, names = [np.ones(len(X))], ["1"]
    for i, n in enumerate(STATE_NAMES):
        cols.append(X[:, i]); names.append(n)
    for i, n in enumerate(["dT", "tau_x", "tau_y", "tau_z"]):
        cols.append(U[:, i]); names.append(n)
    nl = {
        "sin(phi)": sphi, "sin(theta)": sth,
        "sin(theta)cos(phi)": sth * cphi, "cos(phi)cos(theta)-1": cphi * cth - 1,
        "q*sin(phi)*tan(theta)": q * sphi * np.tan(th), "r*cos(phi)*tan(theta)": r * cphi * np.tan(th),
        "q*(cos(phi)-1)": q * (cphi - 1), "r*sin(phi)": r * sphi,
        "p*q": p_ * q, "p*r": p_ * r, "q*r": q * r,
        "dT*sin(theta)cos(phi)": dT * sth * cphi, "dT*sin(phi)": dT * sphi, "dT*(cos(phi)cos(theta)-1)": dT * (cphi * cth - 1),
        "phi^2": phi ** 2, "theta^2": th ** 2, "phi*theta": phi * th,
        "|v|*vx": np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * vx,
        "|v|*vy": np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * vy,
        "|v|*vz": np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * vz,
        "|w|*p": np.sqrt(p_ ** 2 + q ** 2 + r ** 2) * p_, "|w|*q": np.sqrt(p_ ** 2 + q ** 2 + r ** 2) * q,
        "|w|*r": np.sqrt(p_ ** 2 + q ** 2 + r ** 2) * r,
    }
    for n, c in nl.items():
        cols.append(c); names.append(n)
    return np.column_stack(cols), names


def stlsq(Theta, Y, frac=0.01, iters=10):
    """Sequentially thresholded least squares (pinv).  Terms whose contribution is < frac*|target| are removed."""
    scale = np.linalg.norm(Theta, axis=0); scale[scale == 0] = 1.0
    Tn = Theta / scale
    Xi = np.linalg.pinv(Tn) @ Y
    tgt = np.linalg.norm(Y, axis=0)
    for _ in range(iters):
        small = np.abs(Xi) < frac * tgt[None, :]
        Xi[small] = 0.0
        for j in range(Y.shape[1]):
            big = ~small[:, j]
            if big.any():
                Xi[big, j] = np.linalg.pinv(Tn[:, big]) @ Y[:, j]
    return Xi / scale[:, None]


def fit_nonlinear(Xk, Uk, Xn, frac=0.01):
    Y = (Xn - Xk) / CTRL_DT                                    # measured derivative
    Theta, names = build_library(Xk, Uk)
    Xi = stlsq(Theta, Y, frac)
    return Xi, names


def nl_derivative(x, u, Xi):
    Th, _ = build_library(x[None, :], u[None, :])
    return (Th @ Xi)[0]


def print_discovered(Xi, names):
    print("\nDiscovered NONLINEAR sparse dynamics  (x_dot = ...):")
    for j, sn in enumerate(STATE_NAMES):
        terms = [(abs(Xi[i, j]), Xi[i, j], names[i]) for i in range(len(names)) if Xi[i, j] != 0]
        terms.sort(reverse=True)
        txt = " ".join("%+.3g*[%s]" % (c, n) for _, c, n in terms[:6])
        print("  %-5s_dot = %s%s" % (sn, txt, "  ..." if len(terms) > 6 else ""))
    print("  (physics says: p_dot=tau_x/Ixx=%.1f*tau_x, q_dot=%.1f*tau_y, r_dot=%.1f*tau_z, "
          "vx_dot ~ g*sin(theta)cos(phi)=9.81*..., vz_dot ~ (T/m)*(cos cos - 1)...)"
          % (1 / PRM.Ixx, 1 / PRM.Iyy, 1 / PRM.Izz))


def rollouts(Xk, Uk, Xn, model, horizon=60, n_win=150, seed=1):
    """Open-loop multi-step prediction with the recorded inputs. Returns RMSE per state."""
    rng = np.random.default_rng(seed)
    N = len(Xk); errs = []
    # contiguous segments: x_{k+1} of row i must equal x_k of row i+1
    cont = np.r_[np.linalg.norm(Xn[:-1] - Xk[1:], axis=1) < 1e-9]
    starts = [i for i in range(N - horizon) if cont[i:i + horizon - 1].all()]
    pick = rng.choice(starts, size=min(n_win, len(starts)), replace=False)
    for i in pick:
        x = Xk[i].copy(); e = []
        for h in range(horizon):
            x = model(x, Uk[i + h])
            e.append(x - Xn[i + h])
        errs.append(np.array(e))
    errs = np.array(errs)                                      # (win, horizon, 8)
    return np.sqrt(np.mean(errs ** 2, axis=(0, 1))), errs, pick


def experiment_sysid(gui=False):
    print("\n=== PART B: DATA-DRIVEN SYSTEM IDENTIFICATION ===")
    Xk, Uk, Xn = collect_data(duration=90.0, seed=0, gui=gui)
    Xt, Ut, Xtn = collect_data(duration=40.0, seed=7)          # unseen test flight
    print("training samples %d, test samples %d  (dt = %.4f s)" % (len(Xk), len(Xt), CTRL_DT))
    print("max |roll|=%.2f rad  max |pitch|=%.2f rad  -> angles are large enough to see nonlinearity"
          % (np.abs(Xk[:, 0]).max(), np.abs(Xk[:, 1]).max()))

    # ---- linear
    A, B, c = fit_linear(Xk, Uk, Xn)
    print("\nLINEAR model  x_{k+1} = A x_k + B u_k + c   (pseudo-inverse)")
    print("A =\n", A); print("B =\n", B); print("c =", c)
    print("|eigenvalues of A| =", np.round(np.abs(np.linalg.eigvals(A)), 3),
          " (all ~1: open-loop drone is a chain of integrators = marginal; |lambda|>1 would be unstable)")
    lin_model = lambda x, u: A @ x + B @ u + c

    # ---- nonlinear
    Xi, names = fit_nonlinear(Xk, Uk, Xn)
    print_discovered(Xi, names)
    print("active terms: %d of %d x 8" % (np.count_nonzero(Xi), Xi.size))
    nl_model = lambda x, u: x + CTRL_DT * nl_derivative(x, u, Xi)

    # ---- physics
    ph_model = lambda x, u: x + CTRL_DT * physics_derivative(x, u + np.array([PRM.m * G, 0, 0, 0]))

    # ---- evaluation
    models = [("Linear (A,B pinv)", lin_model), ("Nonlinear (SINDy)", nl_model), ("Analytic physics", ph_model)]
    one, multi, ex = {}, {}, {}
    for nm, m in models:
        pred = np.array([m(x, u) for x, u in zip(Xt, Ut)])
        one[nm] = np.sqrt(np.mean((pred - Xtn) ** 2, axis=0))
        multi[nm], errs, pick = rollouts(Xt, Ut, Xtn, m)
    print("\nONE-STEP prediction RMSE on unseen flight")
    print("%-20s" % "model" + "".join("%9s" % s for s in STATE_NAMES))
    for nm in one:
        print("%-20s" % nm + "".join("%9.4f" % v for v in one[nm]))
    print("\n0.5 s OPEN-LOOP prediction RMSE (inputs recorded, states predicted)")
    print("%-20s" % "model" + "".join("%9s" % s for s in STATE_NAMES))
    for nm in multi:
        print("%-20s" % nm + "".join("%9.4f" % v for v in multi[nm]))

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    w = 0.27; xs = np.arange(8)
    for k, nm in enumerate(multi):
        ax[0].bar(xs + (k - 1) * w, one[nm], w, label=nm)
        ax[1].bar(xs + (k - 1) * w, multi[nm], w, label=nm)
    for a, t in zip(ax, ["one-step RMSE", "0.5 s open-loop RMSE"]):
        a.set_xticks(xs); a.set_xticklabels(STATE_NAMES); a.set_yscale("log"); a.set_title(t); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("plots/sysid_rmse.png", dpi=120)

    # example rollout on the largest-angle window
    N = len(Xt); H = 60
    cont = np.r_[np.linalg.norm(Xtn[:-1] - Xt[1:], axis=1) < 1e-9]
    cand = [i for i in range(N - H) if cont[i:i + H - 1].all()]
    i0 = max(cand, key=lambda i: np.abs(Xt[i:i + H, 0]).max() + np.abs(Xt[i:i + H, 1]).max())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, idx in zip(ax, [0, 3, 5]):
        a.plot(np.arange(H) * CTRL_DT, Xtn[i0:i0 + H, idx], "k", lw=2, label="PyBullet (truth)")
        for nm, m in models:
            x = Xt[i0].copy(); tr = []
            for h in range(H):
                x = m(x, Ut[i0 + h]); tr.append(x[idx])
            a.plot(np.arange(H) * CTRL_DT, tr, "--", label=nm)
        a.set_title("predicted %s (open loop)" % STATE_NAMES[idx]); a.set_xlabel("t (s)"); a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("plots/sysid_rollout.png", dpi=120)
    return dict(A=A, B=B, c=c, Xi=Xi, names=names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "pid", "sysid"])
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--opengl2", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    if a.exp in ("all", "pid"):
        experiment_attitude(a.gui)
        experiment_trajectory(a.gui)
    if a.exp in ("all", "sysid"):
        experiment_sysid(False)
    print("\nPlots saved in ./plots")
    if not a.no_show:
        plt.show()