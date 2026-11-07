"""Free system (CI-MPC paper Sec. IV, Fig. 4b): the object is a free 6-dof body
held only by the fingers (no support), commanded to rotate about the vertical axis.
Same costs (Eqs. 3-5), dynamics (Eq. 2) and solver as run_rotz.py; only the
object model, the input bound and the weights differ (planner verification only).

  python run_free.py scene    load/inspect the Free scene, show it in Meshcat
  python run_free.py traj     one DDP trajectory optimization (N=10)
  python run_free.py mpc      closed-loop MPC (+ zero-input baseline, Meshcat)

State x = [q_r (12 hand joints); q_o], q_o = [quaternion wxyz; position xyz] (nx = 19).
"""
import argparse
import os

import numpy as np

import ccp
import mpc
import run_rotz as rz
import scene
from run_rotz import INK, MUTED, SERIES, _fig, _plot_controls

# ============================ CONFIGURATION ==================================
# Everything not listed here is shared with run_rotz.CFG.
W_QUAT, W_POS = 400.0, 1e4            # W_o diagonal: quaternion [1], position [1/m^2]
CFG = dict(
    rz.CFG,
    system="free",                    # free object, no support (scene.py)
    build_dir="build/free",           # generated model files (separate from RotZ)
    # [kg] no support: the fingers carry the whole weight. At 1 kg they must grip;
    # at 0.1 kg smoothed forces alone hold the ball even with every tip > 3 mm away
    obj_mass=1.0,
    hinge_damping=0.5,                # [N m s] object quasi-dynamic damping scale (no hinge here)
    u_max=0.2,                        # [rad] input bound (paper: 0.2 for the Free system)
    omega_ref=0.29,                   # [rad/s] commanded yaw rate (paper's sphere: 0.29)
    # W_o = diag over q_o: a yaw error e changes the quaternion by ~e/2, so
    # W_QUAT = 400 matches W_o = 100 / rad^2 on the yaw in run_rotz.
    w_obj=np.r_[W_QUAT * np.ones(4), W_POS * np.ones(3)],
    mpc_steps=200,                    # closed-loop steps (20 s)
)
NQ = scene.NU  # q_o = x[NQ:]
# =============================================================================


def setup():
    files = scene.write_model_files(CFG)
    s = scene.build_plant(files)
    x0 = scene.initial_grasp(s, CFG)
    return files, s, x0


def q_o_ref(x0, t):
    """Object reference: rotate about world z at omega_ref from the initial
    pose, position held. Rows [qw, qx, qy, qz, px, py, pz]."""
    th = CFG["omega_ref"] * np.asarray(t)
    q = np.zeros((len(th), 7))
    q[:, 0], q[:, 3] = np.cos(th / 2), np.sin(th / 2)  # q_z(th) * q0 with q0 = identity
    q[:, 4:] = x0[NQ + 4:NQ + 7]
    return q


def make_cost(x0, k):
    """Costs for the horizon that starts at MPC step k (t_i = (k + i) h)."""
    t = CFG["h"] * (k + np.arange(CFG["N"] + 1))
    return mpc.make_costs(CFG, x0[:NQ], q_o_ref(x0, t))


def make_problem(files):
    m = ccp.make_model(CFG, files)
    return m, scene.joint_limits(scene.build_plant(files))


def yaw(X):
    """Rotation about world z (twist angle of the quaternion), unwrapped [rad]."""
    X = np.atleast_2d(X)
    return np.unwrap(2 * np.arctan2(X[:, NQ + 3], X[:, NQ]))


def tilt(X):
    """Angle between the object's z axis and world z [rad]."""
    X = np.atleast_2d(X)
    return np.arccos(np.clip(1 - 2 * (X[:, NQ + 1] ** 2 + X[:, NQ + 2] ** 2), -1, 1))


def pos_err(X, x0):
    X = np.atleast_2d(X)
    return np.linalg.norm(X[:, NQ + 4:NQ + 7] - x0[NQ + 4:NQ + 7], axis=1)


def cmd_scene(args):
    files, s, x0 = setup()
    p = s["plant"]
    print("generated model files:", *[f"  {k}: {v}" for k, v in files.items()], sep="\n")
    print(f"\nplant: nq={p.num_positions()} nv={p.num_velocities()}; x = [12 hand joints; "
          f"quaternion wxyz; position xyz], nx = {len(x0)}")
    print(f"x0 object: q = {np.round(x0[NQ:NQ + 4], 4)}, p = {np.round(x0[NQ + 4:], 4)} m")
    print("\nsigned distances hand-object at x0 [mm]:")
    for body, phi in sorted(scene.signed_distances(s, x0)):
        print(f"  {body:10s} {1e3 * phi:+7.2f}")
    if not args.no_meshcat:
        meshcat = rz.start_meshcat()
        sv = scene.build_plant(files, visual=True, meshcat=meshcat)
        scene.show(sv, x0, CFG)
        print(f"\nMeshcat: {meshcat.web_url()}")
        input("press Enter to quit...")


def cmd_traj(args):
    """One short trajectory optimization from zero controls."""
    files, s, x0 = setup()
    m, q_lim = make_problem(files)
    cost = make_cost(x0, 0)
    N, h = CFG["N"], CFG["h"]
    U0 = np.zeros((N, scene.NU))
    X0, _, J0 = mpc.rollout(m, x0, U0, cost, CFG, q_lim)
    X, U, info = mpc.mpc_solve(m, x0, U0, cost, CFG, q_lim, CFG["max_iter_first"])
    print(f"zero controls: cost {J0:.4f} -> {info['costs'][-1]:.4f} in {info['iters']} "
          f"iterations ({info['reason']}), {info['time']:.2f} s")
    print(f"yaw at t=N*h: zero-input {np.degrees(yaw(X0)[-1]):+.3f} deg, optimized "
          f"{np.degrees(yaw(X)[-1]):+.3f} deg (reference {np.degrees(CFG['omega_ref'] * N * h):.1f})")
    print(f"object: max position error {1e3 * pos_err(X, x0).max():.2f} mm, "
          f"max tilt {np.degrees(tilt(X)).max():.2f} deg; max|u| = {np.abs(U).max():.4f} "
          f"<= {CFG['u_max']}; finite: {np.isfinite(X).all() and np.isfinite(U).all()}")
    np.savez(os.path.join(CFG["results_dir"], "free_trajopt.npz"), X=X, U=U, X0=X0,
             costs=info["costs"])


def run_mpc(files, x0, zero_input=False):
    """Closed loop through the same smoothed model (planner verification)."""
    m, q_lim = make_problem(files)
    N, steps = CFG["N"], CFG["mpc_steps"]
    U = np.zeros((N, scene.NU))   # trivial warm start (paper)
    x = x0.copy()
    log = {k: [] for k in ["x", "u", "cost", "iters", "time", "reason", "phi_tip",
                           "f_tip"]}
    for k in range(steps):
        if zero_input:
            u0, info = np.zeros(scene.NU), dict(costs=[np.nan], iters=0, time=0.0, reason="-")
        else:
            iters = CFG["max_iter_first"] if k == 0 else CFG["max_iter_mpc"]
            _, U, info = mpc.mpc_solve(m, x, U, make_cost(x0, k), CFG, q_lim, iters)
            u0 = U[0]
        x_next, contacts = ccp.step(m, x, u0, contacts=True)
        tips = ccp.per_body(contacts)
        log["x"].append(x)
        log["u"].append(u0)
        log["cost"].append(info["costs"][-1])
        log["iters"].append(info["iters"])
        log["time"].append(info["time"])
        log["reason"].append(info["reason"])
        log["phi_tip"].append([tips[t]["phi"] if t in tips else np.nan for t in scene.TIP_FRAMES])
        log["f_tip"].append([np.linalg.norm(tips[t]["f"]) if t in tips else 0.0
                             for t in scene.TIP_FRAMES])
        x = x_next
        if not zero_input:
            U = mpc.shift(U)
    log["x"].append(x)
    out = {k: np.array(v) for k, v in log.items()}
    out["q_lo"], out["q_hi"] = q_lim
    return out


def cmd_mpc(args):
    files, s, x0 = setup()
    h = CFG["h"]
    print("PLANNER VERIFICATION (Free system): MPC actions are executed through the same "
          "smoothed quasi-dynamic CQDC model used for planning.")
    log = run_mpc(files, x0)
    base = run_mpc(files, x0, zero_input=True)
    t = h * np.arange(len(log["x"]))
    th, th_ref = yaw(log["x"]), CFG["omega_ref"] * t
    err = np.degrees(th - th_ref)
    perr, tl = 1e3 * pos_err(log["x"], x0), np.degrees(tilt(log["x"]))
    phi = 1e3 * log["phi_tip"]
    n_contact = np.bincount((phi < 1e3 * CFG["contact_phi"]).sum(1), minlength=4)
    viol = np.maximum(log["x"][:, :NQ] - log["q_hi"], log["q_lo"] - log["x"][:, :NQ])
    print(f"\nyaw: final {np.degrees(th[-1]):.2f} deg, reference {np.degrees(th_ref[-1]):.2f} deg "
          f"({CFG['omega_ref']} rad/s); mean rate {(th[-1] - th[0]) / t[-1]:.3f} rad/s")
    print(f"yaw tracking error: final {err[-1]:+.2f} deg, mean |err| {np.abs(err).mean():.2f} deg, "
          f"max |err| {np.abs(err).max():.2f} deg")
    print(f"object position error: final {perr[-1]:.2f} mm, max {perr.max():.2f} mm; "
          f"max tilt {tl.max():.2f} deg")
    base_drop = x0[NQ + 6] - base["x"][:, NQ + 6]
    print(f"zero-input baseline: final yaw {np.degrees(yaw(base['x'])[-1]):+.3f} deg; "
          + (f"ball dropped (fell {base_drop[-1]:.2f} m; below the hand after "
             f"{h * np.argmax(base_drop > CFG['obj_radius']):.1f} s)" if base_drop.max() > CFG["obj_radius"]
             else f"position error {1e3 * pos_err(base['x'], x0)[-1]:.2f} mm"))
    print(f"fingertips within {1e3 * CFG['contact_phi']:.0f} mm per step: "
          + ", ".join(f"{i}: {n}" for i, n in enumerate(n_contact)))
    print(f"max |u| = {np.abs(log['u']).max():.4f} (bound {CFG['u_max']}); max joint-limit "
          f"violation = {max(viol.max(), 0):.2e} rad; finite: {np.isfinite(log['x']).all()}")
    print(f"DDP iterations per step: mean {log['iters'][1:].mean():.1f}, max {log['iters'].max()}; "
          f"solve time mean {1e3 * log['time'][1:].mean():.0f} ms, max {1e3 * log['time'].max():.0f} ms")
    reasons, counts = np.unique(log["reason"], return_counts=True)
    print("termination reasons:", dict(zip(reasons, counts.tolist())))
    for j in range(3):
        print(f"  tip {j + 1}: " + "".join("#" if v < 1e3 * CFG["contact_phi"] else "."
                                           for v in phi[:, j]))
    np.savez(os.path.join(CFG["results_dir"], "free_mpc.npz"),
             **{f"mpc_{k}": v for k, v in log.items()},
             **{f"base_{k}": v for k, v in base.items()})

    plt, fig, ax = _fig(3, 4, w=14, h=9)
    ax[0].plot(t, np.degrees(th), color=SERIES[0], label="MPC")
    ax[0].plot(t, np.degrees(yaw(base["x"])), color=SERIES[1], label="zero input")
    ax[0].plot(t, np.degrees(th_ref), color=MUTED, ls="--", lw=1, label="reference")
    ax[0].set_title("object yaw [deg]", color=INK, loc="left")
    ax[0].legend()
    ax[1].plot(t, err, color=SERIES[0])
    ax[1].set_title("yaw tracking error [deg]", color=INK, loc="left")
    for i, lab in enumerate("xyz"):
        ax[2].plot(t, 1e3 * (log["x"][:, NQ + 4 + i] - x0[NQ + 4 + i]), color=SERIES[i], label=lab)
    ax[2].set_title("object position error [mm]", color=INK, loc="left")
    ax[2].legend()
    ax[3].plot(t, tl, color=SERIES[0])
    ax[3].set_title("object tilt [deg]", color=INK, loc="left")
    for f in range(3):
        ax[4].plot(t[:-1], phi[:, f], color=SERIES[f], label=f"tip {f + 1}")
        ax[5].plot(t[:-1], log["f_tip"][:, f], color=SERIES[f], label=f"tip {f + 1}")
    ax[4].axhline(1e3 * CFG["contact_phi"], color=MUTED, ls="--", lw=1)
    ax[4].set_title("tip signed distance [mm]", color=INK, loc="left")
    ax[4].legend()
    ax[5].set_title("tip contact force |f| [N] (smoothed)", color=INK, loc="left")
    ax[6].plot(t, 1e3 * (log["x"][:, NQ + 6] - x0[NQ + 6]), color=SERIES[0])
    ax[6].set_title("object height change [mm]", color=INK, loc="left")
    ax[7].plot(t[:-1], log["cost"], color=SERIES[0])
    ax[7].set_yscale("log")
    ax[7].set_title("optimized DDP cost", color=INK, loc="left")
    ax[8].bar(t[:-1], log["iters"], width=0.8 * h, color=SERIES[0])
    ax[8].set_title("DDP iterations per step", color=INK, loc="left")
    ax[9].plot(t[:-1], 1e3 * log["time"], color=SERIES[0])
    ax[9].set_title("solve time [ms]", color=INK, loc="left")
    for a, j in ((ax[10], 1), (ax[11], 2)):
        for f in range(3):
            a.plot(t, log["x"][:, 4 * f + j], color=SERIES[f], label=f"finger {f + 1}")
        a.set_title(f"joint {j + 1} position [rad]", color=INK, loc="left")
    ax[10].legend()
    for a in ax:
        a.set_xlabel("time [s]")
    fig.suptitle("Free system, planner verification: MPC executed through the same smoothed "
                 "quasi-dynamic model (kappa = %g)" % CFG["kappa"], color=INK)
    path = os.path.join(CFG["results_dir"], "free_mpc.png")
    fig.savefig(path, dpi=130)
    print("saved", path)
    if not args.no_meshcat:
        meshcat = rz.start_meshcat()
        sv = scene.build_plant(files, visual=True, meshcat=meshcat)
        scene.animate(sv, log["x"], h, CFG)
        print(f"Meshcat animation: {meshcat.web_url()}  (use the Animation controls to replay)\n"
              f"contact dots (tip within {1e3 * CFG['contact_phi']:.0f} mm): "
              "finger 1 magenta, finger 2 orange, finger 3 yellow")
        input("press Enter to quit...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scene", "traj", "mpc"])
    ap.add_argument("--no-meshcat", action="store_true", help="skip Meshcat")
    args = ap.parse_args()
    os.makedirs(CFG["results_dir"], exist_ok=True)
    {"scene": cmd_scene, "traj": cmd_traj, "mpc": cmd_mpc}[args.command](args)


if __name__ == "__main__":
    main()
