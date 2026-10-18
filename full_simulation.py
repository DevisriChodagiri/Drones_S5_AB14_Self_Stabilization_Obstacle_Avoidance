"""
full_simulation.py   (FILE 4 of 4)  -  COMPLETE SYSTEM in PyBullet
=================================================================
Combines the other three files:
    physics_model.py            -> quadrotor physics (motors, mixer, drag, wind)
    self_stabilizing_pid_ml.py  -> cascaded PID (velocity loop -> attitude loop -> motors)
    obstacle_avoidance_pgmpy.py -> features + risk model (pgmpy BN / LogReg) + decision + avoidance

Control architecture (every 1/120 s):
    obstacles + drone state --> risk model --> SAFE/WARNING/DANGER --> CONTINUE/SLOW_DOWN/AVOID
        --> velocity command v_cmd --> velocity PID --> desired roll/pitch --> attitude PID
        --> torques --> mixer --> 4 motor thrusts --> PyBullet physics (+ optional wind)

Run :  python full_simulation.py                       (3D PyBullet window: zig-zag, then a crossing obstacle)
       python full_simulation.py --speed 3             (faster crossing obstacle in 3D)
       python full_simulation.py --headless            (no window: zig-zag + speed sweep with metrics & plots)
       python full_simulation.py --model lr            (use Logistic Regression instead of the Bayesian net)
       python full_simulation.py --wind 0.4            (add a constant 0.4 N wind along +y)
       python full_simulation.py --trials 5 --no-show
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
import pybullet as p

from physics_model import Quadrotor, CTRL_DT
from self_stabilizing_pid_ml import DroneController
from obstacle_avoidance_pgmpy import (build_models, AvoidanceController, make_zigzag, make_crossing,
                                      ACTIONS, RISK, GOAL, START, DRONE_R)

os.makedirs("plots", exist_ok=True)
ALT = 1.0
ACT_COL = ["tab:green", "tab:orange", "tab:red"]


def add_obstacle_bodies(drone, obstacles):
    ids = []
    for o in obstacles:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=o.r, rgbaColor=[0.9, 0.2, 0.2, 0.9], physicsClientId=drone.cid)
        ids.append(p.createMultiBody(0, -1, vis, [o.p0[0], o.p0[1], ALT], physicsClientId=drone.cid))
    return ids


def run_episode(model, obstacles, avoid=True, gui=False, wind=0.0, T=30.0, drone=None):
    """One full flight START -> GOAL at altitude ALT. Returns log dict + metrics."""
    own = drone is None
    drone = drone or Quadrotor(gui=gui, start_pos=(START[0], START[1], ALT))
    drone.reset((START[0], START[1], ALT))
    ctrl = DroneController()
    av = AvoidanceController(model, enabled=avoid)
    ids = add_obstacle_bodies(drone, obstacles)
    if gui:
        p.addUserDebugText("GOAL", [GOAL[0], GOAL[1], ALT + 0.4], [0, 0.5, 0], 2.0, physicsClientId=drone.cid)
        p.addUserDebugText("START", [START[0], START[1], ALT + 0.4], [0, 0, 0], 2.0, physicsClientId=drone.cid)
        g_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0, 0.8, 0, 0.8], physicsClientId=drone.cid)
        ids.append(p.createMultiBody(0, -1, g_vis, [GOAL[0], GOAL[1], ALT], physicsClientId=drone.cid))
        time.sleep(1.0)
    L = {k: [] for k in ["t", "pos", "vel", "roll", "pitch", "action", "dist", "p_safe", "p_warn", "p_danger", "vcmd"]}
    min_clear, collided, t_goal = np.inf, False, np.nan
    wind_f = (0.0, wind, 0.0)
    try:
        for k in range(int(T / CTRL_DT)):
            t = k * CTRL_DT
            s = drone.get_state()
            pos2, vel2 = s["pos"][:2], s["vel"][:2]
            obs_states = [(o.at(t), o.v, o.r) for o in obstacles]
            v_cmd, info = av.command(pos2, vel2, GOAL, obs_states, s["roll"], s["pitch"])
            f, _ = ctrl.update(s, z_ref=ALT, v_cmd=v_cmd)
            drone.step(f, wind_force=wind_f if wind else (0, 0, 0))
            for o, i in zip(obstacles, ids):
                p.resetBasePositionAndOrientation(i, [*o.at(t + CTRL_DT), ALT], [0, 0, 0, 1], physicsClientId=drone.cid)
                c = np.linalg.norm(o.at(t) - pos2) - o.r - DRONE_R
                min_clear = min(min_clear, c)
                collided |= c < 0
            for key, val in [("t", t), ("pos", s["pos"]), ("vel", s["vel"]), ("roll", s["roll"]), ("pitch", s["pitch"]),
                             ("action", info["action"]), ("dist", info["dist"]), ("p_safe", info["probs"][0]),
                             ("p_warn", info["probs"][1]), ("p_danger", info["probs"][2]), ("vcmd", v_cmd)]:
                L[key].append(val)
            if np.linalg.norm(pos2 - GOAL) < 0.3:
                t_goal = t
                break
            if abs(s["roll"]) > 1.2 or abs(s["pitch"]) > 1.2 or s["pos"][2] < 0.1:
                break                                                   # crashed / flipped
    except p.error:
        print('3-D window was closed during the flight - plotting what was flown so far')
    for i in ids:
        try:
            p.removeBody(i, physicsClientId=drone.cid)
        except p.error:
            pass
    if own:
        drone.close()
    L = {k: np.array(v) for k, v in L.items()}
    alt_err = np.abs(L["pos"][:, 2] - ALT)
    metrics = dict(collided=bool(collided), reached=bool(not np.isnan(t_goal)), min_clearance=float(min_clear),
                   time=float(t_goal) if not np.isnan(t_goal) else np.nan,
                   max_tilt_deg=float(np.degrees(max(np.abs(L["roll"]).max(), np.abs(L["pitch"]).max()))),
                   rms_tilt_deg=float(np.degrees(np.sqrt(np.mean(L["roll"] ** 2 + L["pitch"] ** 2)))),
                   max_alt_err=float(alt_err.max()))
    return L, metrics


def plot_episode(L, obstacles, title, fname):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(2, 3, (1, 4))
    for o in obstacles:
        ts = np.linspace(0, L["t"][-1], 40)
        pts = np.array([o.at(t) for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], ":", c="gray")
        ax.add_patch(plt.Circle(o.at(0), o.r, color="gray", alpha=.3))
        ax.add_patch(plt.Circle(o.at(L["t"][-1]), o.r, color="gray", alpha=.7))
    for a in range(3):
        m = L["action"] == a
        ax.scatter(L["pos"][m, 0], L["pos"][m, 1], s=6, c=ACT_COL[a], label=ACTIONS[a])
    ax.plot(*START, "ko"); ax.plot(*GOAL, "k*", ms=12)
    ax.set_aspect("equal"); ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    a2 = fig.add_subplot(2, 3, 2)
    a2.plot(L["t"], np.degrees(L["roll"]), label="roll"); a2.plot(L["t"], np.degrees(L["pitch"]), label="pitch")
    a2.set_title("attitude (deg)"); a2.legend(); a2.grid(alpha=.3)
    a3 = fig.add_subplot(2, 3, 3)
    a3.plot(L["t"], np.linalg.norm(L["vel"][:, :2], axis=1), label="|v|"); a3.plot(L["t"], np.linalg.norm(L["vcmd"], axis=1), "--", label="|v_cmd|")
    a3.plot(L["t"], L["pos"][:, 2], label="altitude z"); a3.set_title("speed (m/s) & altitude"); a3.legend(); a3.grid(alpha=.3)
    a4 = fig.add_subplot(2, 3, 5)
    a4.stackplot(L["t"], L["p_safe"], L["p_warn"], L["p_danger"], colors=ACT_COL, labels=RISK, alpha=.8)
    a4.set_title("P(SAFE / WARNING / DANGER) of the most threatening obstacle"); a4.legend(loc="upper right", fontsize=7)
    a5 = fig.add_subplot(2, 3, 6)
    a5.step(L["t"], L["action"], where="post"); a5.set_yticks([0, 1, 2]); a5.set_yticklabels(ACTIONS)
    a5.set_title("decision controller output"); a5.grid(alpha=.3)
    for a in (a2, a3, a4, a5):
        a.set_xlabel("t (s)")
    plt.tight_layout(); plt.savefig(fname, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="(default) 3D PyBullet window: zig-zag + crossing obstacle, real time")
    ap.add_argument("--headless", action="store_true", help="no 3D window: zig-zag + speed sweep with metrics")
    ap.add_argument("--sweep", action="store_true", help="same as --headless")
    ap.add_argument("--speed", type=float, default=2.0, help="crossing-obstacle speed for the 3D demo (m/s)")
    ap.add_argument("--seed", type=int, default=3, help="random seed of the 3D crossing demo")
    ap.add_argument("--model", default="bn", choices=["bn", "lr"])
    ap.add_argument("--trials", type=int, default=6, help="random crossing scenarios per obstacle speed")
    ap.add_argument("--wind", type=float, default=0.0, help="constant wind force along +y (N)")
    ap.add_argument("--opengl2", action="store_true", help="use if the 3-D window is blank")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    a.gui = not (a.headless or a.sweep or a.no_show)      # 3-D window is the DEFAULT

    models, _ = build_models(verbose=False)
    model = models.get(a.model, models["lr"])
    print("Risk model:", model.name, "| wind = %.2f N" % a.wind)

    # ---------------- 3-D PyBullet window: watch the drone (real time)
    if a.gui:
        drone = Quadrotor(gui=True, start_pos=(START[0], START[1], ALT))
        print("\n[3D] Scenario 1: zig-zag between 3 obstacles (orange line = flown path)")
        obs = make_zigzag()
        L, m = run_episode(model, obs, avoid=True, wind=a.wind, drone=drone, gui=True)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
        plot_episode(L, obs, "Full system: zig-zag (%s)" % model.name, "plots/full_zigzag.png")
        time.sleep(1.5)
        print("\n[3D] Scenario 2: obstacle crossing the path at %.1f m/s" % a.speed)
        obs2 = make_crossing(a.speed, np.random.default_rng(a.seed))
        L2, m2 = run_episode(model, obs2, avoid=True, wind=a.wind, drone=drone, gui=True)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m2.items()})
        plot_episode(L2, obs2, "Crossing obstacle %.1f m/s" % a.speed, "plots/full_fast_obstacle.png")
        print("\nplots saved in:", os.path.abspath("plots"), "(full_zigzag.png, full_fast_obstacle.png)")
        plt.show(block=False); plt.pause(1.0)                      # plots appear now, 3-D window stays open
        input("\nPress ENTER here to close the 3D window ...")
        drone.close()
        plt.show()
        sys.exit(0)

    # ---------------- 1) zig-zag, three obstacles (headless)
    obs = make_zigzag()
    L, m = run_episode(model, obs, avoid=True, gui=False, wind=a.wind)
    print("\n=== ZIG-ZAG (3 obstacles) ===")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    plot_episode(L, obs, "Full system: zig-zag (%s)" % model.name, "plots/full_zigzag.png")

    # ---------------- 2) obstacles moving at increasing speed (quantitative)
    speeds = [0.5, 1.0, 2.0, 3.0]
    drone = Quadrotor(gui=False, start_pos=(START[0], START[1], ALT))
    rows, example = [], None
    for sp in speeds:
        for mode, en in [("no avoidance", False), ("with avoidance", True)]:
            rng = np.random.default_rng(100 + int(sp * 10))                 # same scenarios for both modes
            ms = []
            for tr in range(a.trials):
                ob = make_crossing(sp, rng)
                Lx, mx = run_episode(model, ob, avoid=en, wind=a.wind, T=25.0, drone=drone)
                ms.append(mx)
                if en and sp == speeds[-1] and tr == 0:
                    example = (Lx, ob)
            df = pd.DataFrame(ms)
            rows.append(dict(speed=sp, mode=mode, collision_rate=df["collided"].mean(),
                             success_rate=(df["reached"] & ~df["collided"]).mean(),
                             min_clearance=df["min_clearance"].mean(), time_s=df["time"].mean(),
                             max_tilt_deg=df["max_tilt_deg"].max(), rms_tilt_deg=df["rms_tilt_deg"].mean(),
                             max_alt_err_m=df["max_alt_err"].max()))
    drone.close()
    res = pd.DataFrame(rows)
    print("\n=== OBSTACLE SPEED SWEEP in PyBullet (%d scenarios per speed) ===" % a.trials)
    print(res.to_string(index=False, float_format=lambda x: "%.2f" % x))
    res.to_csv("plots/full_speed_sweep.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for mode, c in [("no avoidance", "tab:red"), ("with avoidance", "tab:green")]:
        d = res[res["mode"] == mode]
        ax[0].plot(d["speed"], 100 * d["collision_rate"], "o-", c=c, label=mode)
        ax[1].plot(d["speed"], d["min_clearance"], "o-", c=c, label=mode)
        ax[2].plot(d["speed"], d["max_tilt_deg"], "o-", c=c, label=mode)
    for x_, t_, y_ in zip(ax, ["collision rate vs obstacle speed", "mean minimum clearance", "max tilt angle (stability)"],
                          ["%", "m", "deg"]):
        x_.set_title(t_); x_.set_xlabel("obstacle speed (m/s)"); x_.set_ylabel(y_); x_.grid(alpha=.3); x_.legend()
    plt.tight_layout(); plt.savefig("plots/full_speed_sweep.png", dpi=120)
    if example is not None:
        plot_episode(example[0], example[1], "Fast crossing obstacle (%.1f m/s)" % speeds[-1], "plots/full_fast_obstacle.png")
    print("\nPlots + CSV saved in ./plots")
    if not a.no_show:
        plt.show()