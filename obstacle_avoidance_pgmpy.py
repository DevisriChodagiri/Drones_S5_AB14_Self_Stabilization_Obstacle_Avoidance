"""
obstacle_avoidance_pgmpy.py   (FILE 1 of 4)  -  COMPLETE obstacle avoidance in ONE file
=======================================================================================
Run :  python obstacle_avoidance_pgmpy.py             (training, comparison, zig-zag, speed test)
       add --no-show to only save pictures in ./plots

PIPELINE
  1. FEATURES    for every obstacle:   distance d (surface-to-surface), closing speed v_c,
                 time-to-collision  TTC = d / v_c,   |roll|, |pitch| (tilt of the drone)
  2. RISK        two probabilistic classifiers predict  SAFE / WARNING / DANGER
                   (a) Logistic Regression  (continuous features)
                   (b) pgmpy Bayesian Network  (discretised features, CPDs learned from data,
                       exact inference with Variable Elimination)
                 Bayesian network structure:
                       Distance ---> TTC <--- ClosingSpeed
                          |           |
                          +--> Risk <-+---- Tilt
  3. DECISION    expected-cost decision theory:
                    action* = argmin_a  sum_r  P(risk=r) * Cost[a][r]
                 actions: CONTINUE / SLOW_DOWN / AVOID
  4. AVOIDANCE   action -> velocity command (goal attraction, slow-down, side-step around obstacle)
  5. EVALUATION  point-mass kinematic test: 3-obstacle zig-zag + crossing obstacles of
                 increasing speed, collision rate / minimum clearance (quantitative).

The same classes are imported by full_simulation.py (PyBullet drone).
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
try:                                                     # pgmpy (new and old class names)
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BNClass
    except ImportError:
        from pgmpy.models import BayesianNetwork as BNClass
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    HAVE_PGMPY = True
except Exception as e:                                   # pragma: no cover
    HAVE_PGMPY = False
    print("WARNING: pgmpy not available (%s) -> pip install pgmpy" % e)

os.makedirs("plots", exist_ok=True)
G = 9.81
RISK = ["SAFE", "WARNING", "DANGER"]
ACTIONS = ["CONTINUE", "SLOW_DOWN", "AVOID"]
# COST[action][risk]    (rows: CONTINUE, SLOW_DOWN, AVOID   columns: SAFE, WARNING, DANGER)
COST = np.array([[0.0, 3.0, 10.0],
                 [1.0, 1.0, 5.0],
                 [3.0, 2.0, 1.0]])
V_CRUISE = 1.5          # m/s
V_AVOID = 1.8
DRONE_R = 0.15          # m


# =====================================================================================
# 1. FEATURES
# =====================================================================================
def compute_features(pos, vel, obs_pos, obs_vel, obs_r, roll=0.0, pitch=0.0, drone_r=DRONE_R):
    """pos/vel/obs_* are 2-D (x,y) numpy arrays."""
    rel = np.asarray(obs_pos, float) - np.asarray(pos, float)
    dist_c = np.linalg.norm(rel) + 1e-9
    d = dist_c - obs_r - drone_r                                  # surface distance
    n = rel / dist_c
    closing = -np.dot(np.asarray(obs_vel, float) - np.asarray(vel, float), n)   # >0 : approaching
    ttc = d / closing if closing > 0.05 else 20.0
    return dict(dist=max(d, 0.0), closing=closing, ttc=min(max(ttc, 0.0), 20.0),
                roll=abs(roll), pitch=abs(pitch))


def tilt_from_accel(acc):
    """small-angle estimate of tilt produced by a horizontal acceleration (used in kinematic test)."""
    return float(np.arctan(acc / G))


# =====================================================================================
# 2. SYNTHETIC TRAINING DATA (physics-based labelling rule + 5 % label noise)
# =====================================================================================
def generate_dataset(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.02, 6.0, n)
    v = rng.uniform(-1.5, 5.0, n)
    roll = rng.uniform(0.0, 0.5, n)
    pitch = rng.uniform(0.0, 0.5, n)
    ttc = np.where(v > 0.05, d / np.maximum(v, 0.05), 20.0).clip(0, 20)
    k = 1.0 + 0.4 * (np.maximum(roll, pitch) > 0.3)               # tilted drone brakes worse -> be careful earlier
    danger = (d < 0.6 * k) | ((ttc < 1.0 * k) & (d < 3.0))
    warn = (d < 1.5 * k) | ((ttc < 2.5 * k) & (d < 4.0))
    y = np.where(danger, 2, np.where(warn, 1, 0))
    flip = rng.random(n) < 0.05                                   # label noise
    y = np.where(flip, np.clip(y + rng.choice([-1, 1], n), 0, 2), y)
    return pd.DataFrame(dict(dist=d, closing=v, ttc=ttc, roll=roll, pitch=pitch, risk=y))


FEATS = ["dist", "closing", "ttc", "roll", "pitch"]


# =====================================================================================
# 3a. LOGISTIC REGRESSION RISK MODEL
# =====================================================================================
class LogisticRiskModel:
    name = "LogReg"

    def fit(self, df):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=5.0))
        self.clf.fit(df[FEATS].values, df["risk"].values)
        return self

    def predict_proba(self, f):
        x = np.array([[f[k] for k in FEATS]])
        return self.clf.predict_proba(x)[0]

    def predict_df(self, df):
        return self.clf.predict(df[FEATS].values)


# =====================================================================================
# 3b. pgmpy BAYESIAN NETWORK RISK MODEL
# =====================================================================================
def discretise(f):
    dist = "Near" if f["dist"] < 0.8 else ("Mid" if f["dist"] < 2.0 else "Far")
    spd = "Low" if f["closing"] < 0.2 else ("Med" if f["closing"] < 1.2 else "High")
    ttc = "Short" if f["ttc"] < 1.2 else ("Medium" if f["ttc"] < 3.0 else "Long")
    tilt = "Tilted" if max(f["roll"], f["pitch"]) >= 0.3 else "Level"
    return dist, spd, ttc, tilt


class BayesNetRiskModel:
    name = "pgmpy-BN"

    def fit(self, df):
        rows = [discretise(r) for r in df[["dist", "closing", "ttc", "roll", "pitch"]].to_dict("records")]
        data = pd.DataFrame(rows, columns=["Distance", "ClosingSpeed", "TTC", "Tilt"])
        data["Risk"] = [RISK[i] for i in df["risk"].values]
        self.model = BNClass([("Distance", "TTC"), ("ClosingSpeed", "TTC"),
                              ("Distance", "Risk"), ("TTC", "Risk"), ("Tilt", "Risk")])
        # learn every CPD from data (Bayesian estimation with BDeu prior) - works on old and new pgmpy
        est = BayesianEstimator(self.model, data)
        cpds = [est.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=5) for node in self.model.nodes()]
        self.model.add_cpds(*cpds)
        self.model.check_model()
        self.infer = VariableElimination(self.model)
        # exact inference for all 54 evidence combinations -> lookup table (fast in the control loop)
        self.table = {}
        for D in ["Near", "Mid", "Far"]:
            for S in ["Low", "Med", "High"]:
                for T in ["Short", "Medium", "Long"]:
                    for L in ["Level", "Tilted"]:
                        q = self.infer.query(["Risk"], evidence=dict(Distance=D, ClosingSpeed=S, TTC=T, Tilt=L),
                                             show_progress=False)
                        st = q.state_names["Risk"]
                        self.table[(D, S, T, L)] = np.array([q.values[st.index(r)] for r in RISK])
        return self

    def predict_proba(self, f):
        return self.table[discretise(f)]

    def predict_df(self, df):
        out = []
        for r in df[["dist", "closing", "ttc", "roll", "pitch"]].to_dict("records"):
            out.append(int(np.argmax(self.predict_proba(r))))
        return np.array(out)

    def print_structure(self):
        print("Bayesian network edges :", list(self.model.edges()))
        cpd = self.model.get_cpds("Risk")
        print("CPD of Risk has shape", cpd.values.shape, "(Risk | Distance, TTC, Tilt) - learned from data")


def build_models(n=20000, seed=0, verbose=True):
    """Train both models. Returns dict name->model and held-out test dataframe."""
    df = generate_dataset(n, seed)
    tr, te = train_test_split(df, test_size=0.25, random_state=seed)
    models = {"lr": LogisticRiskModel().fit(tr)}
    if HAVE_PGMPY:
        models["bn"] = BayesNetRiskModel().fit(tr)
    if verbose:
        print("\n=== RISK CLASSIFIER COMPARISON (held-out data) ===")
        for k, m in models.items():
            pred = m.predict_df(te)
            print("%-9s accuracy = %.3f" % (m.name, accuracy_score(te["risk"], pred)))
            print(confusion_matrix(te["risk"], pred))
        if "bn" in models:
            models["bn"].print_structure()
    return models, te


# =====================================================================================
# 4. DECISION CONTROLLER  (expected-cost)
# =====================================================================================
def decide(probs):
    """action index minimising expected cost."""
    return int(np.argmin(COST @ probs))


# =====================================================================================
# 5. AVOIDANCE LOGIC  (action -> velocity command)
# =====================================================================================
class Obstacle:
    def __init__(self, pos, vel=(0.0, 0.0), r=0.4):
        self.p0 = np.array(pos, float)
        self.v = np.array(vel, float)
        self.r = r

    def at(self, t):
        return self.p0 + self.v * t


class AvoidanceController:
    def __init__(self, risk_model, v_cruise=V_CRUISE, v_max=2.0, sense_range=6.0, enabled=True, repulse=0.0, repulse_range=1.6):
        self.model, self.v_cruise, self.v_max = risk_model, v_cruise, v_max
        self.repulse, self.repulse_range = repulse, repulse_range      # optional potential-field push away from close obstacles
        self.sense_range, self.enabled = sense_range, enabled
        self.side = {}                       # remembered passing side per obstacle (hysteresis)
        self.origin = None                   # first position (defines the start->goal line)

    def reset(self):
        self.side = {}
        self.origin = None

    def command(self, pos, vel, goal, obs_states, roll=0.0, pitch=0.0):
        """risk-based command (+ optional repulsive push away from every close obstacle, keeps drone centred in corridors)."""
        v, info = self._command_core(pos, vel, goal, obs_states, roll, pitch)
        if self.enabled and self.repulse > 0:
            push = np.zeros(2)
            for (op, ov, orad) in obs_states:
                rel = np.asarray(pos, float) - np.asarray(op, float)
                dc = np.linalg.norm(rel) + 1e-9
                ds = dc - orad - DRONE_R
                if ds < self.repulse_range:
                    push += self.repulse * (1.0 / max(ds, 0.15) - 1.0 / self.repulse_range) * rel / dc
            v = v + push
            sp = np.linalg.norm(v)
            if sp > self.v_max:
                v = v / sp * self.v_max
        return v, info

    def _command_core(self, pos, vel, goal, obs_states, roll=0.0, pitch=0.0):
        """obs_states: list of (pos2, vel2, radius).  Returns v_cmd(2,), info dict."""
        pos, vel, goal = np.asarray(pos, float), np.asarray(vel, float), np.asarray(goal, float)
        to_goal = goal - pos
        dg = np.linalg.norm(to_goal) + 1e-9
        v_pref = self.v_cruise * to_goal / dg * min(1.0, dg / 1.0)
        info = dict(action=0, probs=np.array([1.0, 0, 0]), dist=np.inf, ttc=20.0, idx=-1)
        if not self.enabled:
            return v_pref, info
        best, best_key = None, None
        for i, (op, ov, orad) in enumerate(obs_states):
            f = compute_features(pos, vel, op, ov, orad, roll, pitch)
            if f["dist"] > self.sense_range:
                self.side.pop(i, None)
                continue
            pr = self.model.predict_proba(f)
            a = decide(pr)
            key = (a, -f["dist"])
            if best_key is None or key > best_key:
                best_key, best = key, (i, f, pr, a, np.asarray(op, float), np.asarray(ov, float))
        if best is None:
            return v_pref, info
        i, f, pr, a, op, ov = best
        info.update(action=a, probs=pr, dist=f["dist"], ttc=f["ttc"], idx=i)
        if a == 0:
            return v_pref, info
        n = (op - pos) / (np.linalg.norm(op - pos) + 1e-9)               # unit vector to obstacle
        hv = v_pref / (np.linalg.norm(v_pref) + 1e-9)                    # heading
        perp = np.array([-hv[1], hv[0]])                                 # left of heading
        if self.origin is None:
            self.origin = pos.copy()
        if i not in self.side:                                           # choose passing side once (hysteresis)
            lateral = hv[0] * (op - pos)[1] - hv[1] * (op - pos)[0]      # >0 : obstacle on the left
            lat_pref = -np.sign(lateral) if abs(lateral) > 0.05 else (-np.sign(np.dot(perp, ov)) if abs(np.dot(perp, ov)) > 1e-3 else 1.0)
            line = goal - self.origin
            line = line / (np.linalg.norm(line) + 1e-9)
            best_s, best_score = lat_pref, -np.inf
            for sgn in (1.0, -1.0):                                      # probe the free space on each side
                probe = pos + sgn * perp * 1.8 + hv * 2.0
                clear = min([np.linalg.norm(probe - np.asarray(o2[0], float)) - o2[2] - DRONE_R for o2 in obs_states])
                cross_track = abs(line[0] * (probe - self.origin)[1] - line[1] * (probe - self.origin)[0])
                score = min(clear, 2.5) - 0.3 * cross_track + (0.5 if sgn == lat_pref else 0.0)
                if score > best_score:
                    best_s, best_score = sgn, score
            self.side[i] = best_s
        side = self.side[i]
        if a == 1:                                                       # SLOW_DOWN (+ gentle side-step)
            gain = 0.35 if (f["closing"] > 0.05 and f["dist"] < 1.8) else 0.0     # side-step only if close & approaching
            v = 0.6 * v_pref + gain * side * perp
        else:                                                            # AVOID
            v = V_AVOID * (0.9 * side * perp - 0.3 * n) + 0.4 * v_pref
        sp = np.linalg.norm(v)
        if sp > self.v_max:
            v = v / sp * self.v_max
        return v, info


# =====================================================================================
# SCENARIOS (also used by full_simulation.py)
# =====================================================================================
def make_zigzag():
    """three obstacles slightly off the straight line -> drone must weave left/right."""
    return [Obstacle((3.0, 0.40), r=0.4), Obstacle((6.0, -1.10), r=0.4), Obstacle((9.0, 0.50), r=0.4)]


def make_crossing(speed, rng, v_cruise=V_CRUISE):
    """one obstacle crossing the straight path (y=0) at x_c with the given speed, timed to meet the drone."""
    xc = rng.uniform(3.0, 9.0)
    t_arrive = xc / v_cruise + rng.uniform(-0.5, 0.5)
    direction = rng.choice([-1.0, 1.0])
    y0 = -direction * speed * t_arrive
    return [Obstacle((xc, y0), vel=(0.0, direction * speed), r=0.3)]


START = np.array([0.0, 0.0])
GOAL = np.array([12.0, 0.0])


# =====================================================================================
# KINEMATIC TEST (point-mass drone with first-order velocity response)
# =====================================================================================
def run_kinematic(obstacles, ctrl, T=25.0, dt=0.02, tau=0.35):
    ctrl.reset()
    pos, vel = START.copy(), np.zeros(2)
    traj, acts, min_clear, collided, t_goal = [], [], np.inf, False, np.nan
    roll = pitch = 0.0
    for k in range(int(T / dt)):
        t = k * dt
        obs = [(o.at(t), o.v, o.r) for o in obstacles]
        v_cmd, info = ctrl.command(pos, vel, GOAL, obs, roll, pitch)
        acc = (v_cmd - vel) / tau
        acc_n = np.linalg.norm(acc)
        if acc_n > 4.0:
            acc = acc / acc_n * 4.0
        vel = vel + acc * dt
        pos = pos + vel * dt
        pitch = tilt_from_accel(acc[0]); roll = tilt_from_accel(-acc[1])
        for o in obstacles:
            c = np.linalg.norm(o.at(t) - pos) - o.r - DRONE_R
            min_clear = min(min_clear, c)
            if c < 0:
                collided = True
        traj.append(pos.copy()); acts.append(info["action"])
        if np.linalg.norm(pos - GOAL) < 0.3:
            t_goal = t
            break
    return dict(traj=np.array(traj), acts=np.array(acts), min_clear=min_clear, collided=collided, t_goal=t_goal)


def evaluate_speed_sweep(model, speeds=(0.5, 1.0, 2.0, 3.0, 4.0), trials=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in speeds:
        for mode, en in [("no avoidance", False), ("with avoidance", True)]:
            ctrl = AvoidanceController(model, enabled=en)
            col, clr, tg, ok = 0, [], [], 0
            rng2 = np.random.default_rng(seed + int(s * 100))      # same scenarios for both modes
            for _ in range(trials):
                r = run_kinematic(make_crossing(s, rng2), ctrl)
                col += r["collided"]; clr.append(r["min_clear"]); ok += (not r["collided"]) and not np.isnan(r["t_goal"])
                if not np.isnan(r["t_goal"]):
                    tg.append(r["t_goal"])
            rows.append(dict(speed=s, mode=mode, collision_rate=col / trials, success_rate=ok / trials,
                             mean_min_clearance=float(np.mean(clr)), mean_time=float(np.mean(tg)) if tg else np.nan))
    return pd.DataFrame(rows)


# =====================================================================================
# MAIN
# =====================================================================================
ACT_COL = ["tab:green", "tab:orange", "tab:red"]


def plot_traj(ax, res, obstacles, title):
    tr, ac = res["traj"], res["acts"]
    for o in obstacles:
        ax.add_patch(plt.Circle(o.p0, o.r, color="gray", alpha=.6))
    for a in range(3):
        m = ac == a
        ax.scatter(tr[m, 0], tr[m, 1], s=6, c=ACT_COL[a], label=ACTIONS[a])
    ax.plot(*START, "ko"); ax.plot(*GOAL, "k*", ms=12)
    ax.set_aspect("equal"); ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bn", choices=["bn", "lr"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    models, te = build_models()
    key = a.model if a.model in models else "lr"
    model = models[key]
    print("\nUsing risk model:", model.name)

    # ---- zig-zag
    fig, axs = plt.subplots(1, len(models), figsize=(7 * len(models), 3.8))
    axs = np.atleast_1d(axs)
    for ax, (k, m) in zip(axs, models.items()):
        obs = make_zigzag()
        res = run_kinematic(obs, AvoidanceController(m))
        plot_traj(ax, res, obs, "%s | zig-zag | min clearance %.2f m | %s" %
                  (m.name, res["min_clear"], "COLLISION" if res["collided"] else "no collision"))
        print("zig-zag %-9s min clearance %.2f m, collided=%s, time %.1f s" %
              (m.name, res["min_clear"], res["collided"], res["t_goal"]))
    plt.tight_layout(); plt.savefig("plots/avoid_zigzag.png", dpi=120)

    # ---- speed sweep (quantitative)
    print("\n=== OBSTACLE SPEED SWEEP (%d random crossing scenarios per speed, model=%s) ===" % (a.trials, model.name))
    df = evaluate_speed_sweep(model, trials=a.trials)
    print(df.to_string(index=False, float_format=lambda x: "%.2f" % x))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mode, c in [("no avoidance", "tab:red"), ("with avoidance", "tab:green")]:
        d = df[df["mode"] == mode]
        ax[0].plot(d["speed"], 100 * d["collision_rate"], "o-", c=c, label=mode)
        ax[1].plot(d["speed"], d["mean_min_clearance"], "o-", c=c, label=mode)
    ax[0].set_xlabel("obstacle speed (m/s)"); ax[0].set_ylabel("collision rate (%)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("obstacle speed (m/s)"); ax[1].set_ylabel("mean min clearance (m)"); ax[1].grid(alpha=.3)
    ax[0].set_title("Collision rate vs obstacle speed"); ax[1].set_title("Clearance vs obstacle speed")
    plt.tight_layout(); plt.savefig("plots/avoid_speed_sweep.png", dpi=120)
    print("\nPlots saved in ./plots")
    if not a.no_show:
        plt.show()
