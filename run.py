"""CLI for the contact-implicit MPC prototype (planner verification only).

  python run.py scene      Step 1: load/inspect the scene, show it in Meshcat
  python run.py step       Step 2: verify single smoothed contact-model steps
  python run.py grad       Step 3: check f_x, f_u against finite differences
  python run.py traj       Step 4: one DDP trajectory optimization (N=10)
  python run.py mpc        Step 5: closed-loop MPC (+ baseline, repro check, Meshcat)
"""
import argparse
import os

import numpy as np
from pydrake.all import Role

import ccp
import mpc
import scene

# ============================ CONFIGURATION ==================================
CFG = dict(
    # --- files ---
    urdf="dg3fm/dg3f_m.urdf",         # official Tesollo URDF (read-only)
    mesh_dir="dg3fm/meshes",
    build_dir="build",                # generated model files
    results_dir="results",            # logs and plots
    meshcat_port=7123,                # fixed port on localhost (7000 = macOS AirPlay)
    # --- object: sphere on a world-fixed revolute joint about +z ---
    obj_radius=0.04,                  # [m]
    obj_center=[0.0, 0.0, 0.21],      # [m] above the base housing (top z=0.136)
    obj_mass=0.1,                     # [kg] (only enters via unactuated_mass_scale)
    mu=1.0,                           # hand-object friction coefficient
    # --- hand collision spheres: link suffix -> [(distance along finger, radius)]
    hand_spheres={"3": [(0.012, 0.0125), (0.034, 0.0125)],
                  "4": [(0.016, 0.0112)],
                  "tip": [(0.0175, 0.011)]},
    # --- initial grasp (j1 aligns each finger plane with the object axis) ---
    finger_azimuth=[0.0, 2 * np.pi / 3, -2 * np.pi / 3],
    grasp_j1=[0.0, -np.pi / 3, np.pi / 3],
    grasp_gap=0.001,                  # [m] tip-sphere-to-object gap
    grasp_elevation=0.0,              # [rad] contact point above the equator
    grasp_ik_guess=[1.2, 0.6],        # (j3, j4) initial guess
    # --- smoothed CQDC contact model (paper eq. 2; ccp.py) ---
    h=0.1,                            # [s] time step (paper: 0.1)
    kappa=100.0,                      # log-barrier weight (paper: 100; smaller = smoother)
    Kp=100.0,                         # [N m/rad] joint stiffness (qsim Allegro default)
    hinge_damping=0.5,                # [N m s/rad] object hinge (quasi-dynamic damper)
    contact_detection_tolerance=0.025,  # [m] pairs farther apart are ignored
    # --- MPC / control-limited DDP (mpc.py) ---
    N=10,                             # horizon (paper: 10)
    u_max=0.05,                       # [rad] |joint increment| per step (paper RotZ: 0.05)
    theta_goal=np.radians(20.0),      # [rad] commanded object rotation
    w_obj=100.0,                      # W_o  [1/rad^2]
    w_posture=0.1,                    # W_r = w_posture * I  (nominal = initial grasp)
    w_posture_terminal_scale=1.0,     # W_r multiplier in l_f (paper eq. 3: 1)
    w_u=1.0,                          # W_u = w_u * I
    mu_init=1e-3,                     # initial Levenberg-Marquardt regularization
    tol_step=1e-6,                    # stop if max |feedforward step| below this
    tol_rel_cost=1e-5,                # stop if relative cost decrease below this
    max_iter_first=100,               # DDP iterations, first (cold-start) solve
    max_iter_mpc=20,                  # DDP iterations per MPC step (warm-started)
    mpc_steps=40,                     # closed-loop steps (40 * 0.1 s = 4 s)
    contact_phi=0.003,                # [m] "in contact" threshold (paper: 3 mm)
)
# =============================================================================


def start_meshcat():
    """Meshcat bound explicitly to localhost:<meshcat_port>. Binding to the
    loopback address (not Drake's default wildcard) makes a port clash fail
    loudly instead of being shadowed by another forwarder on 127.0.0.1."""
    from pydrake.geometry import Meshcat, MeshcatParams
    return Meshcat(MeshcatParams(host="localhost", port=CFG["meshcat_port"]))


def setup():
    files = scene.write_model_files(CFG)
    s = scene.build_plant(files)
    x0 = scene.initial_grasp(s, CFG)
    return files, s, x0


def cmd_scene(args):
    files, s, x0 = setup()
    p = s["plant"]
    print("generated model files:", *[f"  {k}: {v}" for k, v in files.items()], sep="\n")
    print(f"\nplant: nq={p.num_positions()} nv={p.num_velocities()} "
          f"actuators={p.num_actuators()} (hand={p.num_actuators(s['hand'])}, "
          f"object={p.num_actuators(s['obj'])})")
    lo, hi = scene.joint_limits(s)
    print("\nstate x (index: joint, axis, limits [rad], x0):")
    for i, j in enumerate(scene.HAND_JOINTS):
        jt = p.GetJointByName(j, s["hand"])
        print(f"  x[{i:2d}] {j:9s} axis={jt.revolute_axis()} "
              f"[{lo[i]:+.3f}, {hi[i]:+.3f}]  x0={x0[i]:+.4f}")
    jo = p.GetJointByName(scene.OBJECT_JOINT, s["obj"])
    print(f"  x[12] {scene.OBJECT_JOINT} axis={jo.revolute_axis()} unlimited  x0={x0[12]:+.4f}")
    assert len(x0) == scene.NX == len(scene.HAND_JOINTS) + 1
    print(f"state dimension nx = {scene.NX} (12 hand + 1 object) verified")
    print("\nfingertip frames (world position at x0):")
    scene.set_x(s, x0)
    for f, name in zip(scene.FINGERS, scene.TIP_FRAMES):
        X = p.CalcRelativeTransform(s["pctx"], p.world_frame(), p.GetFrameByName(name))
        c = scene.tip_sphere_center(s, f, CFG)
        print(f"  {name}: origin={np.round(X.translation(), 4)}  tip-sphere center="
              f"{np.round(c, 4)}  azimuth={np.degrees(np.arctan2(c[1], c[0])):+.1f} deg")
    print("\ncollision geometry (planning model):")
    insp = s["sg"].model_inspector()
    for g in insp.GetAllGeometryIds(Role.kProximity):
        body = p.GetBodyFromFrameId(insp.GetFrameId(g)).name()
        print(f"  {body:12s} {insp.GetShape(g)}  p_BG={np.round(insp.GetPoseInFrame(g).translation(), 4)}")
    print("\nsigned distances hand-object at x0 [mm]:")
    for body, phi in sorted(scene.signed_distances(s, x0)):
        print(f"  {body:10s} {1e3 * phi:+7.2f}")
    print("\nclearance object <-> official collision meshes at x0 [mm] "
          "(vertex-based, negative = penetration):")
    for link, d in scene.mesh_clearance(s, CFG, x0).items():
        print(f"  {link:10s} {1e3 * d:+7.2f}")
    if not args.no_meshcat:
        meshcat = start_meshcat()
        sv = scene.build_plant(files, visual=True, meshcat=meshcat)
        scene.show(sv, x0)
        print(f"\nMeshcat: {meshcat.web_url()}  (toggle 'visual'/'collision' in the menu)")
        input("press Enter to quit...")


def print_contacts(contacts, center):
    for c in sorted(contacts, key=lambda c: c["body"]):
        if c["phi"] < 0.01:
            print(f"      {c['body']:10s} phi={1e3 * c['phi']:+6.2f} mm  "
                  f"|f|={np.linalg.norm(c['f']):6.2f} N  "
                  f"n.f={-np.dot(c['f'], _outward(c['p'], center)):+6.2f} N  "
                  f"tau_z={ccp.object_torque(c, center):+.4f} N m")


def _outward(p, center):
    d = np.asarray(p) - np.asarray(center)
    return d / np.linalg.norm(d)


def cmd_step(args):
    """Step 2: one smoothed CQDC step near contact."""
    files, s, x0 = setup()
    m = ccp.make_model(CFG, files)
    c = CFG["obj_center"]
    print("input: u = joint increments (paper); qsim receives q_cmd = x[:12] + u")
    print(f"h={m.h} s, kappa={m.params.log_barrier_weight}, Kp={CFG['Kp']} N m/rad, "
          f"unactuated_mass_scale={m.params.unactuated_mass_scale:.1f}")

    # (a) qsim's internal plant agrees with the pydrake scene (ordering + geometry).
    _, contacts = ccp.step(m, x0, np.zeros(scene.NU), contacts=True)
    phi_qsim = sorted((c_["body"], round(c_["phi"], 9)) for c_ in contacts)
    phi_drake = sorted((b, round(d, 9)) for b, d in scene.signed_distances(s, x0)
                       if d <= CFG["contact_detection_tolerance"])
    print(f"\n(a) qsim contact pairs == pydrake signed distances at x0: {phi_qsim == phi_drake}"
          f" ({len(phi_qsim)} pairs within tolerance)")

    def J(name):
        return scene.HAND_JOINTS.index(name)
    tests = {}
    u = np.zeros(scene.NU); tests["zero input"] = (x0, u)
    u = np.zeros(scene.NU); u[[J(f"j_dg_{f}_3") for f in scene.FINGERS]] = 0.02
    tests["squeeze (j3 += 0.02)"] = (x0, u)
    for sign in (+1, -1):
        u = tests["squeeze (j3 += 0.02)"][1].copy()
        u[[J(f"j_dg_{f}_1") for f in scene.FINGERS]] = sign * 0.03
        tests[f"squeeze + tangential j1 {sign * 0.03:+.2f}"] = (x0, u)
    print("\n(b) single steps (dtheta = object rotation; contacts with phi < 10 mm):")
    results = {}
    for name, (x, u) in tests.items():
        x1, contacts = ccp.step(m, x, u, contacts=True)
        x1_exact = ccp.step(m, x, u, exact=True)
        results[name] = x1[12] - x[12]
        dq = x1[:12] - x[:12]
        print(f"  {name:32s} finite={np.isfinite(x1).all()}  "
              f"dtheta={np.degrees(x1[12] - x[12]):+8.4f} deg "
              f"(exact SOCP: {np.degrees(x1_exact[12] - x[12]):+8.4f})  "
              f"|dq_hand - u|={np.linalg.norm(dq - u):.2e} rad")
        print_contacts(contacts, c)
    plus, minus = (results[f"squeeze + tangential j1 {v:+.2f}"] for v in (0.03, -0.03))
    print(f"\nopposite tangential pushes -> opposite rotations: "
          f"{np.sign(plus) == -np.sign(minus) != 0}  ({np.degrees(plus):+.3f} / "
          f"{np.degrees(minus):+.3f} deg)")

    # (c) Commands away from contact: rotation per step for j1 = +0.05 (no squeeze)
    # vs. tip-object gap. Nonzero values are the smoothed model's
    # force-at-a-distance; it vanishes as kappa grows / for the exact SOCP and is
    # cut off completely beyond contact_detection_tolerance.
    kappas = [CFG["kappa"], 1e3, 1e4]
    print("\n(c) away from contact: object rotation [deg] for one step with all j1 += 0.05")
    print("    gap [mm] " + "".join(f"  kappa={k:<8g}" for k in kappas) + "  exact SOCP")
    u = np.zeros(scene.NU); u[[J(f"j_dg_{f}_1") for f in scene.FINGERS]] = 0.05
    for gap in [0.001, 0.003, 0.005, 0.010, 0.017, 0.024, 0.030]:
        x = scene.initial_grasp(s, dict(CFG, grasp_gap=gap))
        row = [np.degrees(ccp.step(m, x, u, kappa=k)[12]) for k in kappas]
        row.append(np.degrees(ccp.step(m, x, u, exact=True)[12]))
        print(f"    {1e3 * gap:6.0f}   " + "".join(f"  {v:+12.4f}  " for v in row))


def cmd_grad(args):
    """Step 3: qsim's f_x, f_u (implicit-function derivatives of the smoothed
    CCP) vs. central finite differences of f."""
    files, s, x0 = setup()
    m = ccp.make_model(CFG, files)

    def J(name):
        return scene.HAND_JOINTS.index(name)
    push = np.zeros(scene.NU)
    push[[J(f"j_dg_{f}_3") for f in scene.FINGERS]] = 0.02
    push[[J(f"j_dg_{f}_1") for f in scene.FINGERS]] = 0.03
    x_rot = x0.copy(); x_rot[12] = 0.2; x_rot[[J("j_dg_1_1"), J("j_dg_2_4")]] += [0.1, -0.05]
    cases = {
        "grasp x0, u=0 (gap 1 mm)": (x0, np.zeros(scene.NU)),
        "grasp x0, squeeze+push": (x0, push),
        "touching (gap 0), squeeze+push": (scene.initial_grasp(s, dict(CFG, grasp_gap=0.0)), push),
        "slight penetration (-0.5 mm), u=0": (scene.initial_grasp(s, dict(CFG, grasp_gap=-5e-4)), np.zeros(scene.NU)),
        "gap 5 mm, closing (j3 += 0.05)": (scene.initial_grasp(s, dict(CFG, grasp_gap=0.005)),
                                          0.05 * np.isin(np.arange(12), [J(f"j_dg_{f}_3") for f in scene.FINGERS])),
        "asymmetric, theta=0.2 rad": (x_rot, 0.5 * push),
    }
    epss = [1e-3, 1e-4, 1e-5, 1e-6]
    print("relative error ||J_qsim - J_fd||_F / ||J_fd||_F  (central differences, step eps)")
    print(f"{'case':38s} {'Jac':4s}" + "".join(f"  eps={e:<7g}" for e in epss)
          + "  | obj-row abs err (best eps)")
    for name, (x, u) in cases.items():
        _, fx, fu = ccp.step(m, x, u, grad=True)
        for label, Jq, perturb in (("f_x", fx, "x"), ("f_u", fu, "u")):
            errs, obj = [], []
            for eps in epss:
                n = len(x) if perturb == "x" else len(u)
                Jfd = np.zeros_like(Jq)
                for i in range(n):
                    d = np.zeros(n); d[i] = eps
                    if perturb == "x":
                        Jfd[:, i] = (ccp.step(m, x + d, u) - ccp.step(m, x - d, u)) / (2 * eps)
                    else:
                        Jfd[:, i] = (ccp.step(m, x, u + d) - ccp.step(m, x, u - d)) / (2 * eps)
                errs.append(np.linalg.norm(Jq - Jfd) / np.linalg.norm(Jfd))
                obj.append(np.abs(Jq[12] - Jfd[12]).max())
            b = int(np.argmin(errs))
            print(f"{name:38s} {label:4s}" + "".join(f"  {e:11.2e}" for e in errs)
                  + f"  | {obj[b]:.1e} (|row|max {np.abs(Jq[12]).max():.2f})")


# ---------------------------- plotting ---------------------------------------
# Validated categorical slots in fixed order (finger 1, 2, 3), muted ink for
# references, thin lines, recessive grid, one y-axis per panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, INK, GRID = "#8a8985", "#2b2b29", "#e4e3df"


def _fig(rows, cols, w=11, h=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "lines.linewidth": 2,
                         "font.size": 9, "legend.frameon": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), constrained_layout=True)
    return plt, fig, np.atleast_1d(axes).ravel()


def _plot_controls(axes, t, U, cfg):
    for j, ax in enumerate(axes):
        for f in range(3):
            ax.step(t, U[:, 4 * f + j], where="post", color=SERIES[f], label=f"finger {f + 1}")
        for b in (-cfg["u_max"], cfg["u_max"]):
            ax.axhline(b, color=MUTED, ls="--", lw=1)
        ax.set_title(f"u: joint {j + 1} increments [rad]", color=INK, loc="left")
        ax.set_xlabel("time [s]")
    axes[0].legend(loc="lower left")


def make_problem(files, x0):
    m = ccp.make_model(CFG, files)
    cost = mpc.make_costs(CFG, x0[:scene.NU], CFG["theta_goal"])
    s = scene.build_plant(files)
    return m, cost, scene.joint_limits(s)


def cmd_traj(args):
    """Step 4: one short trajectory optimization from zero controls."""
    files, s, x0 = setup()
    m, cost, q_lim = make_problem(files, x0)
    N, h = CFG["N"], CFG["h"]
    U0 = np.zeros((N, scene.NU))  # initialization: zero controls (no other guess)
    X0, _, J0 = mpc.rollout(m, x0, U0, cost, CFG, q_lim)
    X, U, info = mpc.mpc_solve(m, x0, U0, cost, CFG, q_lim, CFG["max_iter_first"])
    lo_ok = all(np.all(U[i] >= mpc.input_bounds(CFG, X[i], *q_lim)[0] - 1e-12) and
                np.all(U[i] <= mpc.input_bounds(CFG, X[i], *q_lim)[1] + 1e-12) for i in range(N))
    print(f"initialization: zero controls; cost {J0:.4f} -> {info['costs'][-1]:.4f} "
          f"in {info['iters']} iterations ({info['reason']}), {info['time']:.2f} s")
    print(f"object angle at t=N*h: zero-input {np.degrees(X0[-1, 12]):+.3f} deg, "
          f"optimized {np.degrees(X[-1, 12]):+.3f} deg (goal {np.degrees(CFG['theta_goal']):.1f})")
    print(f"finite: X {np.isfinite(X).all()}, U {np.isfinite(U).all()}; max|u| = "
          f"{np.abs(U).max():.4f} <= {CFG['u_max']}; input+command bounds respected: {lo_ok}")
    lo, hi = q_lim
    viol = np.maximum(X[:, :12] - hi, lo - X[:, :12]).max()
    print(f"max joint-limit violation along X: {max(viol, 0):.2e} rad")
    plt, fig, ax = _fig(2, 4, w=13, h=6)
    ax[0].plot(info["costs"], color=SERIES[0], marker="o", ms=4)
    ax[0].set_title("DDP cost", color=INK, loc="left"); ax[0].set_xlabel("iteration")
    ax[0].set_yscale("log")
    t = h * np.arange(N + 1)
    ax[1].plot(t, np.degrees(X0[:, 12]), color=MUTED, label="initial (u = 0)")
    ax[1].plot(t, np.degrees(X[:, 12]), color=SERIES[0], label="optimized")
    ax[1].axhline(np.degrees(CFG["theta_goal"]), color=MUTED, ls="--", lw=1, label="goal")
    ax[1].set_title("object angle [deg]", color=INK, loc="left"); ax[1].set_xlabel("time [s]")
    ax[1].legend()
    for a in (ax[2], ax[3]):
        a.axis("off")
    ax[2].text(0, 0.9, f"N = {N}, h = {h} s, kappa = {CFG['kappa']}\n"
               f"init: zero controls\niterations: {info['iters']} ({info['reason']})\n"
               f"cost: {J0:.3f} -> {info['costs'][-1]:.3f}\nsolve time: {info['time']:.2f} s",
               va="top", color=INK, transform=ax[2].transAxes)
    _plot_controls(ax[4:], h * np.arange(N), U, CFG)
    path = os.path.join(CFG["results_dir"], "step4_trajopt.png")
    fig.savefig(path, dpi=130)
    print("saved", path)
    np.savez(os.path.join(CFG["results_dir"], "step4_trajopt.npz"), X=X, U=U, X0=X0,
             costs=info["costs"])


def run_mpc(files, x0, zero_input=False, exact_plant=False):
    """Closed loop: solve, apply u_0 to the SAME smoothed quasi-dynamic model
    (planner verification, not physical validation), observe, shift, replan.
    exact_plant=True executes u_0 in the unsmoothed SOCP instead (diagnostic)."""
    m, cost, q_lim = make_problem(files, x0)
    c = CFG["obj_center"]
    N, steps = CFG["N"], CFG["mpc_steps"]
    U = np.zeros((N, scene.NU))   # trivial warm start (paper); no other init
    x = x0.copy()
    log = {k: [] for k in ["x", "u", "cost", "iters", "time", "phi_tip", "f_tip",
                           "tau", "tau_contact", "reason"]}
    for k in range(steps):
        if zero_input:
            u0, info = np.zeros(scene.NU), dict(costs=[np.nan], iters=0, time=0.0, reason="-")
        else:
            iters = CFG["max_iter_first"] if k == 0 else CFG["max_iter_mpc"]
            _, U, info = mpc.mpc_solve(m, x, U, cost, CFG, q_lim, iters)
            u0 = U[0]
        x_next, contacts = ccp.step(m, x, u0, contacts=True, exact=exact_plant)
        tips = {cc["body"]: cc for cc in contacts if cc["body"].endswith("tip")}
        log["x"].append(x); log["u"].append(u0)
        log["cost"].append(info["costs"][-1]); log["iters"].append(info["iters"])
        log["time"].append(info["time"]); log["reason"].append(info["reason"])
        log["phi_tip"].append([tips[t]["phi"] if t in tips else np.nan for t in scene.TIP_FRAMES])
        log["f_tip"].append([np.linalg.norm(tips[t]["f"]) if t in tips else 0.0
                             for t in scene.TIP_FRAMES])
        log["tau"].append(sum(ccp.object_torque(cc, c) for cc in contacts))
        log["tau_contact"].append(sum(ccp.object_torque(cc, c) for cc in contacts
                                      if cc["phi"] < CFG["contact_phi"]))
        x = x_next
        if not zero_input:
            U = mpc.shift(U)
    log["x"].append(x)
    out = {k: np.array(v) for k, v in log.items()}
    out["q_lo"], out["q_hi"] = q_lim
    return out


def cmd_mpc(args):
    """Step 5: closed-loop MPC + zero-input baseline + reproducibility check."""
    files, s, x0 = setup()
    h, goal = CFG["h"], CFG["theta_goal"]
    if args.plant == "exact":
        log = run_mpc(files, x0, exact_plant=True)
        th = log["x"][:, 12]
        print("DIAGNOSTIC: plan with the smoothed model (kappa = %g), execute u_0 in the "
              "exact (unsmoothed) CQDC SOCP. Still quasi-dynamic, not a physics simulation."
              % CFG["kappa"])
        print("t [s]  theta [deg]  min tip phi [mm]")
        for k in range(0, len(th), 5):
            phi = np.nanmin(log["phi_tip"][min(k, len(th) - 2)])
            print(f"{k * h:5.1f}  {np.degrees(th[k]):+10.3f}  {1e3 * phi:8.2f}")
        print(f"final angle {np.degrees(th[-1]):+.3f} deg (goal {np.degrees(goal):.1f}); "
              f"final error {np.degrees(mpc.wrap(th[-1] - goal)):+.3f} deg")
        np.savez(os.path.join(CFG["results_dir"], "diag_exact_plant.npz"),
                 **{k: v for k, v in log.items()})
        return
    print("PLANNER VERIFICATION: MPC actions are executed through the same smoothed "
          "quasi-dynamic CQDC model used for planning (not a physical simulation).")
    log = run_mpc(files, x0)
    base = run_mpc(files, x0, zero_input=True)
    rep = run_mpc(files, x0)
    t = h * np.arange(len(log["x"]))
    th = log["x"][:, 12]
    err = np.degrees(mpc.wrap(th - goal))
    viol = np.maximum(log["x"][:, :12] - log["q_hi"], log["q_lo"] - log["x"][:, :12])
    first = np.argmax(np.abs(err) < 5.0) if (np.abs(err) < 5.0).any() else None
    print(f"\nobject angle: start {np.degrees(th[0]):.2f} deg, final {np.degrees(th[-1]):.3f} deg, "
          f"goal {np.degrees(goal):.1f} deg")
    print(f"tracking error: initial {err[0]:+.2f} deg -> final {err[-1]:+.3f} deg; "
          f"max |err| over last 1 s {np.abs(err[-int(1 / h):]).max():.3f} deg; "
          f"first |err| < 5 deg at t = {f'{t[first]:.1f}' if first is not None else 'never'} s")
    print(f"zero-input baseline final angle: {np.degrees(base['x'][-1, 12]):+.4f} deg "
          f"(final error {np.degrees(mpc.wrap(base['x'][-1, 12] - goal)):+.2f} deg)")
    print(f"finite: x {np.isfinite(log['x']).all()}, u {np.isfinite(log['u']).all()}, "
          f"cost {np.isfinite(log['cost']).all()}")
    print(f"max |u| = {np.abs(log['u']).max():.4f} (bound {CFG['u_max']}); "
          f"max joint-limit violation = {max(viol.max(), 0):.2e} rad")
    print(f"DDP iterations per step: mean {log['iters'][1:].mean():.1f}, max {log['iters'].max()} "
          f"(first solve {log['iters'][0]}); solve time mean {1e3 * log['time'][1:].mean():.1f} ms, "
          f"max {1e3 * log['time'].max():.1f} ms")
    reasons, counts = np.unique(log["reason"], return_counts=True)
    print("termination reasons:", dict(zip(reasons, counts.tolist())))
    dth = np.abs(np.diff(th))
    in_contact = np.nanmin(log["phi_tip"], axis=1) < CFG["contact_phi"]
    near, far = np.abs(log["tau_contact"]).sum(), np.abs(log["tau"] - log["tau_contact"]).sum()
    print(f"contact: {100 * dth[in_contact].sum() / dth.sum():.2f}% of total |dtheta| accrued in "
          f"steps with a tip at phi < {1e3 * CFG['contact_phi']:.0f} mm; pairs with phi >= "
          f"{1e3 * CFG['contact_phi']:.0f} mm carry {100 * far / (near + far):.1f}% of sum |tau|")
    print(f"tip phi range [{1e3 * np.nanmin(log['phi_tip']):.2f}, "
          f"{1e3 * np.nanmax(log['phi_tip']):.2f}] mm; max tip force {log['f_tip'].max():.1f} N")
    dx = max(np.abs(log["x"] - rep["x"]).max(), np.abs(log["u"] - rep["u"]).max())
    print(f"reproducibility: max |difference| between two runs = {dx:.1e}")
    np.savez(os.path.join(CFG["results_dir"], "step5_mpc.npz"),
             **{f"mpc_{k}": v for k, v in log.items()},
             **{f"base_{k}": v for k, v in base.items()})

    plt, fig, ax = _fig(3, 4, w=14, h=9)
    ax[0].plot(t, np.degrees(th), color=SERIES[0], label="MPC")
    ax[0].plot(t, np.degrees(base["x"][:, 12]), color=SERIES[1], label="zero input")
    ax[0].axhline(np.degrees(goal), color=MUTED, ls="--", lw=1, label="target")
    ax[0].set_title("object angle [deg]", color=INK, loc="left"); ax[0].legend()
    ax[1].plot(t, np.abs(err), color=SERIES[0])
    ax[1].axhline(5, color=MUTED, ls="--", lw=1)
    ax[1].set_yscale("log"); ax[1].set_title("|tracking error| [deg]", color=INK, loc="left")
    ax[2].plot(t[:-1], log["cost"], color=SERIES[0])
    ax[2].set_yscale("log"); ax[2].set_title("optimized DDP cost", color=INK, loc="left")
    ax[3].bar(t[:-1], log["iters"], width=0.8 * h, color=SERIES[0])
    ax[3].set_title("DDP iterations per step", color=INK, loc="left")
    for j in range(4):
        a = ax[4 + j]
        for f in range(3):
            i = 4 * f + j
            a.plot(t, log["x"][:, i], color=SERIES[f], label=f"finger {f + 1}")
        for b in sorted(set(np.r_[log["q_lo"][j::4], log["q_hi"][j::4]])):
            if np.abs(b - log["x"][:, j::4]).min() < 0.5:
                a.axhline(b, color=MUTED, ls="--", lw=1)
        a.set_title(f"joint {j + 1} position [rad]", color=INK, loc="left")
    ax[4].legend()
    for f in range(3):
        ax[8].plot(t[:-1], 1e3 * log["phi_tip"][:, f], color=SERIES[f], label=f"tip {f + 1}")
        ax[9].plot(t[:-1], log["f_tip"][:, f], color=SERIES[f], label=f"tip {f + 1}")
    ax[8].axhline(1e3 * CFG["contact_phi"], color=MUTED, ls="--", lw=1)
    ax[8].set_title("tip signed distance [mm]", color=INK, loc="left"); ax[8].legend()
    ax[9].set_title("tip contact force |f| [N] (smoothed)", color=INK, loc="left")
    ax[10].plot(t[:-1], log["tau"], color=SERIES[0], label="all pairs")
    ax[10].plot(t[:-1], log["tau_contact"], color=SERIES[1], ls="--",
                label=f"pairs with phi < {1e3 * CFG['contact_phi']:.0f} mm")
    ax[10].set_title("contact torque on object [N m]", color=INK, loc="left"); ax[10].legend()
    ax[11].plot(t[:-1], 1e3 * log["time"], color=SERIES[0])
    ax[11].set_title("solve time [ms]", color=INK, loc="left")
    for a in ax:
        a.set_xlabel("time [s]")
    fig.suptitle("Planner verification: MPC executed through the same smoothed "
                 "quasi-dynamic model (kappa = %g)" % CFG["kappa"], color=INK)
    path = os.path.join(CFG["results_dir"], "step5_mpc.png")
    fig.savefig(path, dpi=130)
    print("saved", path)
    if not args.no_meshcat:
        meshcat = start_meshcat()
        sv = scene.build_plant(files, visual=True, meshcat=meshcat)
        scene.animate(sv, log["x"], h)
        print(f"Meshcat animation: {meshcat.web_url()}  (use the Animation controls to replay)")
        input("press Enter to quit...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scene", "step", "grad", "traj", "mpc"])
    ap.add_argument("--no-meshcat", action="store_true", help="skip Meshcat")
    ap.add_argument("--plant", choices=["same", "exact"], default="same",
                    help="mpc: execute actions in the planning model (default) or, as a "
                         "diagnostic, in the exact unsmoothed SOCP")
    args = ap.parse_args()
    os.makedirs(CFG["results_dir"], exist_ok=True)
    {"scene": cmd_scene, "step": cmd_step, "grad": cmd_grad, "traj": cmd_traj, "mpc": cmd_mpc}[args.command](args)


if __name__ == "__main__":
    main()
