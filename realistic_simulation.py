"""
realistic_simulation.py   (FILE 5)  -  Drone flying through a realistic WORLD with WIND
=======================================================================================
Scene : a park path lined with TREES, BUILDINGS (with windows) on both sides, a SHED in the corridor,
        a BIRD patrolling across the path, a WINDSOCK, trees that sway in the wind.
Wind  : mean wind + turbulence (Ornstein-Uhlenbeck / Dryden-like) + random gusts (1-cos shape).
        aerodynamic force from RELATIVE air speed:   F = c * ( |v| v  -  |v-w| (v-w) )   [extra to the built-in drag]
Drone : same cascaded PID + physics as the other files; avoidance = risk model (pgmpy / LogReg)
        + expected-cost decision + avoidance velocity (obstacle_avoidance_pgmpy.py).

Run :  python realistic_simulation.py                       3-D window, real time (default)
       python realistic_simulation.py --wind-speed 7        stronger wind
       python realistic_simulation.py --headless            no window, plots only
       python realistic_simulation.py --wind-speed 4        no window: plots + metrics
       python realistic_simulation.py --eval                wind sweep 0 / 2 / 4 / 6 m/s, avoidance vs none
       python realistic_simulation.py --snapshot            save pictures of the world (plots/world_*.png)
Options: --wind-dir 90 (deg, 90 = blowing toward +y = cross wind)  --no-bird  --model lr|bn  --no-show
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pybullet as p

from physics_model import Quadrotor, PRM, CTRL_DT
from self_stabilizing_pid_ml import DroneController
from obstacle_avoidance_pgmpy import build_models, AvoidanceController, Obstacle, ACTIONS, DRONE_R

os.makedirs("plots", exist_ok=True)
ALT = 1.5
START = np.array([0.0, 0.0])
GOAL = np.array([28.0, 0.0])
ACT_COL = ["tab:green", "tab:orange", "tab:red"]


# =====================================================================================
#                                       WIND
# =====================================================================================
class Wind:
    """w(t) = mean*(1 + gust(t)) + turbulence(t)."""

    def __init__(self, speed=4.0, direction_deg=90.0, turb_sigma=0.7, gust_frac=0.6, seed=0):
        self.rng = np.random.default_rng(seed)
        th = np.radians(direction_deg)
        self.speed = speed
        self.mean = speed * np.array([np.cos(th), np.sin(th), 0.0])
        self.sigma = turb_sigma if speed > 0 else 0.0
        self.tau = 1.5                                  # turbulence correlation time (s)
        self.gust_frac = gust_frac
        self.turb = np.zeros(3)
        self.next_gust = self.rng.uniform(3.0, 6.0)
        self.gust = None
        self.w = self.mean.copy()

    def step(self, t, dt=CTRL_DT):
        # Ornstein-Uhlenbeck turbulence:  dx = -x/tau dt + sigma*sqrt(2/tau) dW
        noise = self.rng.standard_normal(3) * np.array([1.0, 1.0, 0.3])
        self.turb += -self.turb / self.tau * dt + self.sigma * np.sqrt(2 * dt / self.tau) * noise
        if self.speed > 0 and t >= self.next_gust:      # start a new gust
            self.gust = (t, self.rng.uniform(2.0, 4.0))
            self.next_gust = t + self.rng.uniform(5.0, 9.0)
        g = 0.0
        if self.gust is not None and t < self.gust[0] + self.gust[1]:
            g = 0.5 * (1.0 - np.cos(2 * np.pi * (t - self.gust[0]) / self.gust[1]))     # 1-cos gust shape
        self.w = self.mean * (1.0 + self.gust_frac * g) + self.turb
        return self.w

    def force(self, v):
        """extra force (on top of the built-in -c|v|v drag) so that total drag uses the relative air velocity."""
        c = PRM.c_drag
        rel = v - self.w
        return c * (np.linalg.norm(v) * v - np.linalg.norm(rel) * rel)


# =====================================================================================
#                                       WORLD
# =====================================================================================
class Patrol(Obstacle):
    """a bird flying back and forth across the path (triangular wave in y)."""

    def __init__(self, x, y_min=-8.0, y_max=8.0, speed=2.2, phase=0.3, r=0.25):
        super().__init__((x, 0.0), (0, 0), r)
        self.x, self.y0, self.y1, self.sp, self.phase = x, y_min, y_max, speed, phase
        self.L = y_max - y_min

    def _s(self, t):
        return (self.phase * self.L + self.sp * t) % (2 * self.L)

    def at(self, t):
        s = self._s(t)
        return np.array([self.x, self.y0 + (s if s <= self.L else 2 * self.L - s)])

    def vel(self, t):
        return np.array([0.0, self.sp if self._s(t) <= self.L else -self.sp])


def obs_vel(o, t):
    return o.vel(t) if hasattr(o, "vel") else o.v


# (cx, cy, half_x, half_y, height, rgb)
BUILDINGS = [(7.0, 5.8, 3.0, 2.0, 7.0, (0.75, 0.55, 0.45)),
             (6.0, -6.2, 2.2, 2.5, 4.5, (0.80, 0.78, 0.70)),
             (16.0, -6.2, 3.5, 2.5, 9.0, (0.55, 0.60, 0.70)),
             (24.5, 5.5, 2.5, 2.0, 5.5, (0.85, 0.70, 0.55)),
             (27.0, -6.0, 2.0, 2.0, 6.0, (0.65, 0.50, 0.50))]
SHED = (21.0, -3.3, 1.1, 1.1, 2.6, (0.55, 0.40, 0.30))
TREES = [(4.0, 0.7, 1.0), (10.0, -1.0, 1.1), (14.0, 1.4, 0.9), (18.0, -0.9, 1.0), (23.0, 1.3, 1.1), (26.0, -1.0, 0.9)]


def _box(cid, pos, half, rgb, alpha=1.0):
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[*rgb, alpha], physicsClientId=cid)
    return p.createMultiBody(0, -1, v, pos, physicsClientId=cid)


def _tree(cid, x, y, s, rng):
    tr = p.createVisualShape(p.GEOM_CYLINDER, radius=0.12 * s, length=1.6 * s, rgbaColor=[0.40, 0.26, 0.13, 1], physicsClientId=cid)
    p.createMultiBody(0, -1, tr, [x, y, 0.8 * s], physicsClientId=cid)
    green = np.array([0.12, 0.45, 0.14]) * rng.uniform(0.8, 1.2)
    cv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75 * s, rgbaColor=[*np.clip(green, 0, 1), 1], physicsClientId=cid)
    top = p.createVisualShape(p.GEOM_SPHERE, radius=0.5 * s, rgbaColor=[*np.clip(green * 1.2, 0, 1), 1], physicsClientId=cid)
    c1 = p.createMultiBody(0, -1, cv, [x, y, 1.9 * s], physicsClientId=cid)
    c2 = p.createMultiBody(0, -1, top, [x, y, 2.55 * s], physicsClientId=cid)
    return [(c1, np.array([x, y, 1.9 * s])), (c2, np.array([x, y, 2.55 * s]))]


def _building(cid, cx, cy, hx, hy, h, rgb):
    _box(cid, [cx, cy, h / 2], [hx, hy, h / 2], rgb)
    _box(cid, [cx, cy, h + 0.12], [hx + 0.15, hy + 0.15, 0.12], (0.25, 0.25, 0.28))              # roof
    face = -1.0 if cy > 0 else 1.0                                                                 # face towards the path
    yw = cy + face * (hy + 0.01)
    floors = max(1, int((h - 1.0) // 2.2))
    cols = max(1, int(2 * hx // 1.6))
    for fl in range(floors):
        for c in range(cols):
            x = cx - hx + (c + 0.5) * (2 * hx / cols)
            _box(cid, [x, yw, 1.4 + fl * 2.2], [0.45, 0.02, 0.5], (0.25, 0.4, 0.6))
    _box(cid, [cx - hx + 0.8, yw, 0.55], [0.4, 0.025, 0.55], (0.35, 0.2, 0.1))                    # door


def building_circles(cx, cy, hx, hy):
    """cover the footprint with circles (conservative) so the 2-D avoidance can handle boxes."""
    r = 1.15 * min(hx, hy)
    n = max(1, int(np.ceil(2 * hx / (1.2 * r))))
    xs = np.linspace(cx - hx + r, cx + hx - r, n) if n > 1 else [cx]
    return [Obstacle((x, cy), r=r) for x in xs]


def build_world(cid, with_bird=True, seed=1, decorative=True):
    is_gui = p.getConnectionInfo(cid)["connectionMethod"] == p.GUI
    if is_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cid)     # much faster world building
        print("building 3-D world ... (a few seconds)")
    world = _build_world(cid, with_bird, seed, decorative)
    if is_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)
        p.resetDebugVisualizerCamera(6.5, 65, -22, [START[0], START[1], ALT], physicsClientId=cid)
        print("world ready - the drone is about to take off")
    return world


def _build_world(cid, with_bird=True, seed=1, decorative=True):
    rng = np.random.default_rng(seed)
    plane_ids = [i for i in range(p.getNumBodies(physicsClientId=cid))]
    p.changeVisualShape(0, -1, rgbaColor=[0.30, 0.55, 0.28, 1], physicsClientId=cid)              # grass
    _box(cid, [14, 0, 0.005], [16, 1.6, 0.005], (0.62, 0.52, 0.38))                                # dirt path
    obstacles, sway = [], []
    for b in BUILDINGS:
        _building(cid, *b)
        obstacles += building_circles(*b[:4])
    _box(cid, [SHED[0], SHED[1], SHED[4] / 2], [SHED[2], SHED[3], SHED[4] / 2], SHED[5])
    _box(cid, [SHED[0], SHED[1], SHED[4] + 0.08], [SHED[2] + 0.2, SHED[3] + 0.2, 0.08], (0.3, 0.1, 0.1))
    obstacles += building_circles(*SHED[:4])
    for (x, y, s) in TREES:
        sway += _tree(cid, x, y, s, rng)
        obstacles.append(Obstacle((x, y), r=0.7 * s))
    if decorative:                                                                                 # background forest
        for _ in range(26):
            x, y = rng.uniform(-3, 33), rng.choice([-1, 1]) * rng.uniform(10, 18)
            _tree(cid, x, y, rng.uniform(0.9, 1.6), rng)
    # windsock
    _box(cid, [-2.0, -3.0, 1.2], [0.03, 0.03, 1.2], (0.8, 0.8, 0.8))
    sock = _box(cid, [-1.8, -3.0, 2.3], [0.3, 0.07, 0.07], (0.95, 0.35, 0.1))
    # goal & start markers
    gv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=0.02, rgbaColor=[0.1, 0.8, 0.2, 0.9], physicsClientId=cid)
    p.createMultiBody(0, -1, gv, [GOAL[0], GOAL[1], 0.02], physicsClientId=cid)
    sv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=0.02, rgbaColor=[0.9, 0.9, 0.2, 0.9], physicsClientId=cid)
    p.createMultiBody(0, -1, sv, [START[0], START[1], 0.02], physicsClientId=cid)
    bird = None
    if with_bird:
        bird = Patrol(14.0)
        bv = p.createVisualShape(p.GEOM_SPHERE, radius=bird.r, rgbaColor=[0.95, 0.95, 0.95, 1], physicsClientId=cid)
        bird.body = p.createMultiBody(0, -1, bv, [*bird.at(0), ALT], physicsClientId=cid)
    return dict(static=obstacles, bird=bird, sway=sway, sock=sock)


def save_snapshots(cid):
    for name, eye, tgt in [("world_perspective", [-6, -13, 7], [14, 0, 1.5]), ("world_topdown", [14, 0.01, 38], [14, 0, 0])]:
        view = p.computeViewMatrix(eye, tgt, [0, 0, 1], physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(60 if "persp" in name else 45, 16 / 9, 0.1, 100, physicsClientId=cid)
        img = p.getCameraImage(1100, 620, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
        rgb = np.reshape(img[2], (620, 1100, 4))[:, :, :3].astype(np.uint8)
        plt.imsave("plots/%s.png" % name, rgb)
        print("saved plots/%s.png" % name)


# =====================================================================================
#                                    EPISODE
# =====================================================================================
def run_episode(model, wind_speed=3.0, wind_dir=90.0, avoid=True, gui=False, with_bird=True, seed=0, T=80.0, drone=None, world=None):
    own = drone is None
    drone = drone or Quadrotor(gui=gui, start_pos=(START[0], START[1], ALT))
    cid = drone.cid
    drone.reset((START[0], START[1], ALT))
    world = world or build_world(cid, with_bird)
    obstacles = list(world["static"]) + ([world["bird"]] if world["bird"] else [])
    wind = Wind(wind_speed, wind_dir, seed=seed)
    ctrl = DroneController(gains=dict(vel_xy=(2.5, 0.6, 0.05)))
    av = AvoidanceController(model, enabled=avoid, sense_range=6.0, repulse=0.35)
    v_f = np.zeros(2)                       # low-pass filtered velocity command (tau = 0.5 s) -> no chattering
    if gui:
        p.resetDebugVisualizerCamera(6.5, 65, -22, [START[0], START[1], ALT], physicsClientId=cid)
        h_arrow = h_txt = h_act = -1
        time.sleep(1.0)
    L = {k: [] for k in ["t", "pos", "vel", "roll", "pitch", "action", "wind", "wspd", "dist"]}
    min_clear, collided, t_goal, hit = np.inf, False, np.nan, None
    try:
        for k in range(int(T / CTRL_DT)):
            t = k * CTRL_DT
            s = drone.get_state()
            pos2, vel2 = s["pos"][:2], s["vel"][:2]
            w = wind.step(t)
            obs_states = [(o.at(t), obs_vel(o, t), o.r) for o in obstacles]
            v_cmd, info = av.command(pos2, vel2, GOAL, obs_states, s["roll"], s["pitch"])
            v_f += (v_cmd - v_f) * CTRL_DT / 0.5
            f, _ = ctrl.update(s, z_ref=ALT, v_cmd=v_f)
            drone.step(f, wind_force=wind.force(s["vel"]))
            for o in obstacles:
                c = np.linalg.norm(o.at(t) - pos2) - o.r - DRONE_R
                min_clear = min(min_clear, c)
                if c < 0 and not collided:
                    hit = o
                collided |= c < 0
            if world["bird"]:
                p.resetBasePositionAndOrientation(world["bird"].body, [*world["bird"].at(t + CTRL_DT), ALT], [0, 0, 0, 1], physicsClientId=cid)
            if gui and k % 4 == 0:
                ws = float(np.linalg.norm(w[:2])); wd = w[:2] / (ws + 1e-9)
                pp = s["pos"]
                h_arrow = p.addUserDebugLine(pp.tolist(), (pp + np.array([wd[0], wd[1], 0]) * min(1.5, 0.35 * ws)).tolist(), [0.1, 0.7, 1.0], 5, 0,
                                             replaceItemUniqueId=h_arrow, physicsClientId=cid)
                h_txt = p.addUserDebugText("wind %.1f m/s | speed %.1f m/s" % (ws, np.linalg.norm(s["vel"][:2])), [pp[0], pp[1], pp[2] + 0.75],
                                           [0.0, 0.2, 0.8], 1.4, 0, replaceItemUniqueId=h_txt, physicsClientId=cid)
                h_act = p.addUserDebugText(ACTIONS[info["action"]], [pp[0], pp[1], pp[2] + 0.55], ACT_COL_RGB[info["action"]], 1.6, 0,
                                           replaceItemUniqueId=h_act, physicsClientId=cid)
                ang = np.arctan2(w[1], w[0])
                p.resetBasePositionAndOrientation(world["sock"], [-2.0 + 0.3 * np.cos(ang), -3.0 + 0.3 * np.sin(ang), 2.3],
                                                  p.getQuaternionFromEuler([0, 0, ang]), physicsClientId=cid)
                for j, (bid, base) in enumerate(world["sway"]):          # trees sway in the wind
                    off = 0.02 * ws * np.sin(2 * np.pi * 0.6 * t + j) + 0.015 * ws
                    p.resetBasePositionAndOrientation(bid, (base + np.array([wd[0], wd[1], 0]) * off).tolist(), [0, 0, 0, 1], physicsClientId=cid)
            for key, val in [("t", t), ("pos", s["pos"]), ("vel", s["vel"]), ("roll", s["roll"]), ("pitch", s["pitch"]),
                             ("action", info["action"]), ("wind", w.copy()), ("wspd", np.linalg.norm(w[:2])), ("dist", info["dist"])]:
                L[key].append(val)
            if np.linalg.norm(pos2 - GOAL) < 0.4:
                t_goal = t
                break
            if abs(s["roll"]) > 1.2 or abs(s["pitch"]) > 1.2 or s["pos"][2] < 0.15:
                break
    except p.error:
        print('3-D window was closed during the flight - plotting what was flown so far')
    if own:
        drone.close()
    L = {k: np.array(v) for k, v in L.items()}
    m = dict(collided=bool(collided), reached=bool(not np.isnan(t_goal)), min_clearance=float(min_clear),
             time=float(t_goal) if not np.isnan(t_goal) else np.nan,
             max_tilt_deg=float(np.degrees(max(np.abs(L["roll"]).max(), np.abs(L["pitch"]).max()))),
             max_alt_err=float(np.abs(L["pos"][:, 2] - ALT).max()), mean_wind=float(L["wspd"].mean()), max_wind=float(L["wspd"].max()))
    if hit is not None:
        print("  collision with obstacle at (%.1f, %.1f) r=%.2f %s" % (*hit.p0, hit.r, "(BIRD)" if isinstance(hit, Patrol) else ""))
    return L, m, world


ACT_COL_RGB = [[0, 0.6, 0], [1, 0.5, 0], [0.9, 0, 0]]


def plot_episode(L, world, title, fname):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, 1, 1)
    for b in BUILDINGS:
        ax.add_patch(Rectangle((b[0] - b[2], b[1] - b[3]), 2 * b[2], 2 * b[3], color="#b5a089"))
    ax.add_patch(Rectangle((SHED[0] - SHED[2], SHED[1] - SHED[3]), 2 * SHED[2], 2 * SHED[3], color="#8a6a50"))
    for (x, y, s) in TREES:
        ax.add_patch(Circle((x, y), 0.7 * s, color="#2f8f3a", alpha=.8))
    for o in world["static"]:
        ax.add_patch(Circle(o.p0, o.r, fill=False, ls=":", ec="gray", lw=.8))
    if world["bird"]:
        ts = np.linspace(0, L["t"][-1], 300); pts = np.array([world["bird"].at(t) for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], ",", c="lightgray")
        ax.add_patch(Circle(world["bird"].at(L["t"][-1]), 0.25, color="k"))
    for a in range(3):
        m = L["action"] == a
        ax.scatter(L["pos"][m, 0], L["pos"][m, 1], s=7, c=ACT_COL[a], label=ACTIONS[a], zorder=5)
    ax.plot(*START, "ko"); ax.plot(*GOAL, "k*", ms=14)
    wmean = L["wind"][:, :2].mean(axis=0)
    ax.annotate("", xy=(2.5, -10 + 3 * wmean[1] / (np.linalg.norm(wmean) + 1e-9)), xytext=(2.5, -10), arrowprops=dict(arrowstyle="->", color="c", lw=3))
    ax.text(3.2, -10.5, "mean wind %.1f m/s" % np.linalg.norm(wmean), color="c")
    ax.set_xlim(-3, 31); ax.set_ylim(-12, 9); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.3)
    a2 = fig.add_subplot(2, 3, 4)
    a2.plot(L["t"], L["wspd"], label="wind speed"); a2.plot(L["t"], np.linalg.norm(L["vel"][:, :2], axis=1), label="drone speed")
    a2.set_title("wind (mean + turbulence + gusts) vs drone speed [m/s]"); a2.legend(); a2.grid(alpha=.3)
    a3 = fig.add_subplot(2, 3, 5)
    a3.plot(L["t"], np.degrees(L["roll"]), label="roll"); a3.plot(L["t"], np.degrees(L["pitch"]), label="pitch")
    a3.set_title("attitude (deg): drone leans into the wind"); a3.legend(); a3.grid(alpha=.3)
    a4 = fig.add_subplot(2, 3, 6)
    a4.plot(L["t"], L["pos"][:, 2]); a4.set_ylim(ALT - 0.3, ALT + 0.3); a4.set_title("altitude (m) - held at %.1f m" % ALT); a4.grid(alpha=.3)
    for a in (a2, a3, a4):
        a.set_xlabel("t (s)")
    plt.tight_layout(); plt.savefig(fname, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="(default) open the 3-D PyBullet window")
    ap.add_argument("--headless", action="store_true", help="no 3-D window, only plots/metrics")
    ap.add_argument("--opengl2", action="store_true", help="use if the 3-D window is blank")
    ap.add_argument("--wind-speed", type=float, default=3.0)
    ap.add_argument("--wind-dir", type=float, default=90.0)
    ap.add_argument("--model", default="lr", choices=["lr", "bn"])
    ap.add_argument("--no-bird", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    # 3-D window is the DEFAULT; headless only with --headless / --no-show / --eval / --snapshot
    a.gui = not (a.headless or a.no_show or a.eval or a.snapshot)

    models, _ = build_models(verbose=False)
    model = models.get(a.model, models["lr"])
    print("Risk model:", model.name, "| wind %.1f m/s from dir %.0f deg | bird: %s" % (a.wind_speed, a.wind_dir, not a.no_bird))

    if a.snapshot:
        d = Quadrotor(gui=False, start_pos=(START[0], START[1], ALT), fancy=True)
        build_world(d.cid, not a.no_bird)
        save_snapshots(d.cid)
        d.close()

    if a.eval:
        rows = []
        drone = Quadrotor(gui=False, start_pos=(START[0], START[1], ALT))
        world = build_world(drone.cid, not a.no_bird)
        for ws in [0.0, 2.0, 4.0, 6.0]:
            for mode, en in [("no avoidance", False), ("with avoidance", True)]:
                ms = [run_episode(model, ws, a.wind_dir, en, False, not a.no_bird, seed=a.seed + i, drone=drone, world=world)[1] for i in range(a.trials)]
                df = pd.DataFrame(ms)
                rows.append(dict(wind=ws, mode=mode, collision_rate=df["collided"].mean(), success_rate=(df["reached"] & ~df["collided"]).mean(),
                                 min_clearance=df["min_clearance"].mean(), time_s=df["time"].mean(), max_tilt_deg=df["max_tilt_deg"].max(),
                                 max_alt_err_m=df["max_alt_err"].max()))
        drone.close()
        res = pd.DataFrame(rows)
        print("\n=== WIND SWEEP in the realistic world (%d runs each) ===" % a.trials)
        print(res.to_string(index=False, float_format=lambda x: "%.2f" % x))
        res.to_csv("plots/realistic_wind_sweep.csv", index=False)
    elif a.gui:
        drone = Quadrotor(gui=True, start_pos=(START[0], START[1], ALT))
        L, m, world = run_episode(model, a.wind_speed, a.wind_dir, True, True, not a.no_bird, a.seed, drone=drone)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        plot_episode(L, world, "Realistic world, wind %.1f m/s (%s)" % (a.wind_speed, model.name), "plots/realistic_flight.png")
        print("\nplots saved in:", os.path.abspath("plots"), "(realistic_flight.png)")
        plt.show(block=False); plt.pause(1.0)                      # plots appear now, 3-D window stays open
        input("\nPress ENTER to close the 3D window ...")
        drone.close()
        plt.show()
    elif not a.snapshot:
        L, m, world = run_episode(model, a.wind_speed, a.wind_dir, True, False, not a.no_bird, a.seed)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        plot_episode(L, world, "Realistic world, wind %.1f m/s (%s)" % (a.wind_speed, model.name), "plots/realistic_flight.png")
        print("saved plots/realistic_flight.png")
        if not a.no_show:
            plt.show()