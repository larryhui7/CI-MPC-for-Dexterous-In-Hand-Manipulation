# Contact-implicit MPC prototype: DG-3F-M constrained rotation (planner verification)

A small reproduction of the **high-level contact-implicit MPC** of Jiang et al.,
*Contact-Implicit Model Predictive Control for Dexterous In-hand Manipulation*
(IROS 2024), on a Tesollo DG-3F-M hand rotating a sphere about a fixed vertical
hinge by 20°.

> **Scope.** This is a *planner verification* experiment. MPC actions are
> executed through the **same smoothed quasi-dynamic model** used for planning.
> It does not validate the method in full physical dynamics, and the paper's
> low-level compliance-based contact controller (Sec. III-B) is **not**
> implemented. It is not a reproduction of the full paper.

## Setup and run

Prerequisites (macOS arm64): `brew install python@3.13 cmake yaml-cpp`.

```bash
./setup.sh                          # venv + Drake 1.58.0 + build qsim_cpp (~1-2 min)
.venv/bin/python run.py scene       # Step 1: inspect the scene, show it in Meshcat
.venv/bin/python run.py step        # Step 2: single smoothed contact-model steps
.venv/bin/python run.py grad        # Step 3: f_x, f_u vs. central finite differences
.venv/bin/python run.py traj        # Step 4: one DDP solve (N = 10), plot
.venv/bin/python run.py mpc         # Step 5: closed-loop MPC + baseline + repro check + Meshcat animation
.venv/bin/python run.py mpc --plant exact   # optional diagnostic (see below)
```

Meshcat is served at **http://localhost:7123** (`CFG["meshcat_port"]`) only while the command is running; it waits for Enter before exiting. The port is fixed because macOS AirPlay takes 7000 and port forwarders (e.g. Colima/Docker) can shadow Drake's default fallback ports. `--no-meshcat` skips the viewer (and the "press Enter" prompt). All parameters
are in the `CONFIGURATION` block at the top of `run.py`. Outputs go to `results/`.

## Files

| File | Purpose |
|---|---|
| `scene.py` | Generates the planning models from the official URDF (sphere collision geometry, object SDF, qsim model directive). Builds a pydrake plant of the same scene for inspection, grasp IK, signed distances and Meshcat visualization/animation. |
| `ccp.py` | Contact-model interface: the paper's eq. (2) CCP, smoothed with a log barrier, solved by `qsim_cpp`. Converts the paper's joint-increment input to qsim's position command, returns `f`, `f_x`, `f_u`, and smoothed contact forces. |
| `mpc.py` | Costs (eqs. 3–5) with exact derivatives, control-limited DDP (box-QP backward pass, LM regularization, line search), joint-limit-aware input bounds, MPC warm-start shift. |
| `run.py` | Configuration block, CLI for Steps 1–5, logging, plots. |
| `setup.sh`, `requirements.txt` | Dependency build/spec. `setup.sh` pins qsim to commit `c917755` and verifies the Drake tarball's SHA-256. |
| `qsim_drake1.58.patch` | Minimal patch so qsim (written for Drake 1.21) builds and runs on Drake 1.58 (see below). |

Generated (git-ignored): `build/models/*` (planning URDF/SDF/directives), `external/` (Drake headers, qsim), `.venv/`.

## Paper → implementation

| Paper | Here |
|---|---|
| State `x = q = [q_r; q_o]` (eq. 1) | `x = [12 hand joints (j_dg_{1,2,3}_{1..4}); object yaw θ]`, `nx = 13` (asserted) |
| Input `u = S_r δq`, `u̲ ≤ u ≤ ū` | `u ∈ R^12` joint increments, `|u_i| ≤ 0.05 rad` (paper's RotZ bound) |
| Dynamics `f` = CQDC (eq. 2), analytically smoothed with κ | `qsim_cpp` `kLogIcecream` (log barrier on the friction cones), `log_barrier_weight = κ = 100` (paper) |
| Input meaning | qsim's input is the **absolute** joint position command `q_cmd` (stiff joint springs `Kp (q_cmd − q_r)`). We use `q_cmd = q_r + u` |
| `f_x, f_u` "from the open-source implementation of [2]" | qsim's implicit-function derivatives `A = ∂q⁺/∂q`, `B = ∂q⁺/∂q_cmd` (GradientMode `kAB`). Chain rule: `f_x = A + B S_r`, `f_u = B` |
| `λ` (dual of eq. 2) | log-barrier implicit dual `λ_i = 2/(κ(w₀²−‖w_t‖²))·[w₀, −w₁, −w₂]`, `w_i = J_i v + [φ_i/(hμ),0,0]` (added in the patch) |
| `C_o = ‖q_o − q_o,ref‖²_Wo` | `w_o · wrap(θ − θ_ref)²` (local angle error, derivative 1) |
| `C_r = ‖q_r − q_r,ref‖²_Wr` | nominal posture = the initial grasp |
| `C_u = ‖u‖²_Wu`, `l_f = C_o + C_r` | as written. Optional terminal `W_r` scale (paper: "increase W_r near the end"), set to 1 |
| Control-limited DDP [Tassa 2014], no `f_xx, f_xu, f_uu` | iLQR with a projected-Newton **box QP in every backward-pass stage** (free-set feedback gains), V_xx regularization, backtracking line search on the clamped closed-loop rollout |
| h = 0.1 s, N = 10 | same |

The paper used Crocoddyl for DDP. Here the DDP is implemented in ~150 lines of NumPy (`mpc.py`).

## Scene

* **Hand.** Official `dg3fm/dg3f_m.urdf` (read-only), base link `l_dg_mount` welded to the world. 12 revolute joints, in `x` order:
  `j_dg_1_1..4, j_dg_2_1..4, j_dg_3_1..4` (axes, limits and x0 are printed by `run.py scene`). Fingertip frames: `l_dg_{1,2,3}_tip`.
* **Actuator mapping.** The URDF has **no `<transmission>` elements, so Drake creates 0 actuators**. qsim treats every DoF of the hand model instance as a position-controlled joint (stiffness `Kp`), consistent with the DG-3F-M's 12 directly driven joints. Joint limits come from the URDF.
* **Object.** A sphere (r = 4 cm, center at z = 0.21 m above the mount, above the 0.136 m-tall base housing) on a revolute joint about world +z, `effort = 0`, so Drake attaches **no actuator**. No torque, position command or assistance is ever applied to it. qsim adds only gravity, which has zero moment about the vertical axis. Object rotation can come only from contact forces. Visual-only stripes make rotation visible.
* **Grasp.** The three fingers are rotated by j1 = (0, −60°, +60°), so each finger plane contains the hinge axis (azimuths 0°, ±120°). j2 = 0, and (j3, j4) come from IK (`scene.initial_grasp`) so each tip sphere is 1 mm outside the object at the equator. Verified: tip-sphere signed distances +1.00 mm, all other hand spheres ≥ 26 mm away. The vertex-based distance from the object to the **official** tip collision meshes is +6.9 mm, i.e. no hidden penetration.

### Collision simplification and exclusions

* Drake 1.58 **cannot load the official `.STL` collision meshes** (its convex-hull path accepts only .obj/.vtk/.gltf), so they must be replaced.
* qsim's `f_x` uses AutoDiff signed distances. In Drake these exist **only for Sphere–{Sphere, Box, HalfSpace}**; capsules and cylinders throw. So every hand collision shape is a sphere and the object is a sphere (the paper also restricts itself to "spheres, capsules").
* Spheres per finger (sized from the STL bounding boxes, `CFG["hand_spheres"]`): 2 on the proximal flexion link `l_dg_*_3` (r 12.5 mm), 1 on `l_dg_*_4` (r 11.2 mm), 1 on the tip `l_dg_*_tip` (r 11 mm, reaching the end of the pad). These are coarse: there are gaps along the links, and the spheres are slightly larger than the real 19 mm link thickness.
* **No collision geometry on the base housing or j1/j2 links.** The object can only rotate about its fixed axis and stays above them.
* **Hand self-collision excluded** (`drake:collision_filter_group`). Only hand–object contact is modeled.
* Official `.dae` visual meshes are kept for Meshcat (`visual/` layer). The collision spheres are shown in the `collision/` layer.

## Dependency: quasistatic_simulator on Drake 1.58

The CQDC model and its derivatives come from `pangtao22/quasistatic_simulator` (C++ `qsim_cpp`, last commit Sep 2023, built for Drake 1.21). It **builds and runs** after `qsim_drake1.58.patch`:

1. **Build.** Modern CMake Python discovery; skip tests/examples (no GTest); C++20.
2. **API drift.** `Eigen::all/last` → `Eigen::placeholders::*`, `RigidBody::is_floating()` → `is_floating_base_body()`, and `ContactResults` is now built from a vector.
3. **Package paths.** Drake no longer ships `manipulation/models`, and the `robotics_utilities` submodule is optional, so both are registered only if they exist.
4. **Zero-contact steps.** With no contact pair inside the detection tolerance, the log-barrier "phase 1" SOCP (`min s`, no cone constraints) is unbounded and qsim threw. Every v is then strictly feasible, so we start Newton from v = 0.
5. **Smoothed contact forces.** qsim computed forces only for the exact modes. The patch adds the log-barrier dual (formula above) and a binding `get_contact_arrays()` returning plain arrays. pydrake 1.58 uses nanobind, so Drake objects (`ContactResults`, plants, `ModelInstanceIndex`) cannot cross qsim's pybind11 boundary.
6. **Newton tolerance.** The barrier Newton stopping tolerance goes from 1e-6 to 1e-12 (Newton decrement). With 1e-6, f was accurate only to ~1e-5 rad, and its derivatives disagreed with finite differences by ~1e-3 relative (see Step 3).

Toolchain notes: `qsim_cpp` is compiled against the Drake 1.58.0 **binary tarball headers** but loads the **pip wheel's `libdrake.so`** (same release, same install name; verified that only one libdrake is loaded). On this machine the Command Line Tools clang could not find libc++ headers. `setup.sh` detects this and points clang at the SDK's copy.

## Results (all numbers from `results/*.txt`, produced by the commands above)

**Step 1 – scene.** `nq = nv = 13`, 0 Drake actuators, state dimension 13 verified. qsim's internal contact pairs equal pydrake's signed distances at x0, which also cross-checks the joint ordering.

**Step 2 – one contact-model step** (h = 0.1 s, κ = 100, Kp = 100 N·m/rad, hinge damping 0.5 N·m·s/rad):

| test | object Δθ (smoothed) | exact SOCP | tip contact forces |
|---|---|---|---|
| u = 0 at grasp (1 mm gap) | 0.0000° | 0.0000° | 12.5 N each, radial (force-at-a-distance), τ_z = 0 |
| squeeze (j3 += 0.02) | 0.0000° | 0.0000° | 24.3 N each, radial, τ_z = 0 |
| squeeze + j1 += 0.03 | **+0.547°** | +0.564° | τ_z = +0.016 N·m per tip |
| squeeze + j1 −= 0.03 | **−0.547°** | −0.564° | τ_z = −0.016 N·m per tip |

Signs and units: φ in m (positive = separated), forces in N, pointing into the object (n·f > 0).

**Away from contact** (all j1 += 0.05, no squeeze, object rotation per step):

| tip gap | κ = 100 (paper) | κ = 1e3 | κ = 1e4 | exact SOCP |
|---|---|---|---|---|
| 1 mm | 0.838° | 0.609° | 0.214° | 0 |
| 10 mm | 0.263° | 0.032° | 0.003° | 0 |
| 24 mm | 0.086° | 0.009° | 0.001° | 0 |
| ≥ 25 mm (detection tolerance) | 0 | 0 | 0 | 0 |

So **with the paper's κ = 100, commands away from contact do move the object**. This is the smoothing's force-at-a-distance, which the paper acknowledges and addresses with its low-level controller. It is not a bug, but it is a real property of the planning model.

**Step 3 – derivatives.** Relative error `‖J_qsim − J_fd‖_F / ‖J_fd‖_F` (central differences):

| case | f_x, ε=1e-3 | f_x, ε=1e-4 | f_x, ε=1e-5 | f_u, ε=1e-5 |
|---|---|---|---|---|
| grasp, u = 0 (gap 1 mm) | 6.2e-5 | 5.9e-7 | 6.1e-8 | 8.0e-7 |
| grasp, squeeze + push | 1.1e-4 | 1.1e-6 | 2.1e-7 | 3.2e-7 |
| touching (gap 0) | 1.1e-4 | 8.9e-7 | 1.7e-7 | 6.7e-8 |
| −0.5 mm penetration | 5.1e-5 | 5.1e-7 | 1.1e-8 | 1.5e-8 |
| gap 5 mm, closing | 2.1e-5 | 2.1e-7 | 2.3e-9 | 3.0e-8 |
| asymmetric, θ = 0.2 | 8.6e-5 | 4.4e-6 | 4.9e-5 | 6.1e-6 |

Errors fall as O(ε²) down to ~1e-8, i.e. qsim's analytic derivatives match. (The asymmetric case's growth at small ε is round-off/solver noise in f.) Before the Newton-tolerance change, the same table plateaued at ~1e-3.

**Step 4 – one trajectory** (`results/step4_trajopt.png`). Zero-control initialization, no other warm start. Cost 134.0 → 18.2 in 11 iterations (0.11 s). Object reaches 19.96° at t = 1 s, versus 0.00° for the u = 0 rollout. Inputs are within ±0.05 and command bounds hold, with no joint-limit violation. The solution squeezes (j3) and moves the tips tangentially with **both** j1 (yaw) and j2 (sideways tilt, ~0.1 m lever).

**Step 5 – closed loop** (`results/step5_mpc.png`, 40 steps = 4 s):

| metric | value |
|---|---|
| object angle | 0° → **19.998°** (target 20°) |
| tracking error | −20° → **−0.002°**; < 5° from t = 0.3 s |
| zero-input baseline | stays at 0.000° (error −20°) |
| finite x, u, cost | yes |
| max \|u\|, joint-limit violation | 0.0500 rad (bound 0.05), 0 rad |
| DDP iterations / solve time | first solve 11; then mean 5.8 (max 20); mean 50 ms, max 164 ms |
| contact | 100% of the rotation happens in steps with a tip at φ < 3 mm (paper's threshold). Tip φ stays in [0.4, 4.9] mm. Pairs at φ ≥ 3 mm carry 37.5% of Σ\|τ\|, mostly balanced holding forces after the target is reached |
| tip forces | up to 99 N (smoothed model, Kp = 100) |
| reproducibility | two runs: max \|ΔX\|, \|ΔU\| = 0 (deterministic); identical after a from-scratch rebuild |

**Diagnostic (`--plant exact`).** Planning uses κ = 100, but each action is executed in qsim's *exact* (unsmoothed) SOCP. The loop still reaches 19.98° (peak overshoot 21.03° at t = 0.4 s), and the tips close to φ ≈ 0 (real contact). This is still the same quasi-dynamic family, **not** a physical simulation.

### Caveats

* Rotation is fast (20° in ~3 steps) because the input bound is per joint and j2's long lever arm moves the tips a lot. The constant 20° set-point reference rewards rushing.
* Forces of 60–100 N exceed what a DG-3F-M (2 N·m joint effort limit) could produce. Kp = 100 matches qsim's Allegro example, so that κ = 100 has its intended meaning; torque limits belong to the low-level layer.
* κ = 100 lets the planner use near-contact (mm-gap) forces. The exact-plant diagnostic suggests replanning absorbs this here, but only a real dynamics simulation can tell.
* Joint limits are enforced by tightening the input bounds so that `q_cmd` stays within limits. **This is an implementation addition**, not in the paper. qsim itself does not enforce joint limits.

## Reproduced vs. simplified

| Component | Status |
|---|---|
| Quasi-dynamic CQDC model (eq. 2), qsim implementation as in the paper | **Reproduced** (same library, patched for Drake 1.58) |
| Analytic smoothing, κ = 100 | **Reproduced** (log-barrier `kLogIcecream`) |
| f_x, f_u from the CQDC implementation | **Reproduced** (qsim `kAB`), checked vs. FD |
| x = q, u = S_r δq, constant input bounds (0.05 rad) | **Reproduced** (explicit `q_cmd = q_r + u` conversion) |
| Costs C_o, C_r, C_u, terminal C_o + C_r | **Reproduced** (weights chosen here; the paper gives none) |
| Control-limited DDP without 2nd-order dynamics terms | **Reproduced** in NumPy (paper: Crocoddyl) |
| Receding horizon N = 10, h = 0.1 s, warm start from shifted solution | **Reproduced** |
| Hand | **Changed**: DG-3F-M (3 fingers, 12 DoF) instead of Allegro (16 DoF) |
| Collision geometry | **Simplified**: spheres only; no self-collision, base or proximal-link contact |
| Object/task | **Simplified**: sphere on a damped vertical hinge, 20° set-point (paper RotZ: continuous rotation with finger gaiting) |
| Joint-limit handling | **Addition**: limit-aware input bounds |
| Smoothed contact-force output | **Addition** (log-barrier dual) |
| Low-level compliance-based contact controller (Sec. III-B, eqs. 6–19) | **Not implemented** |
| Execution in full dynamics (Drake hydroelastic sim) | **Not done**: planner verification only |
| Finger gaiting, free object, disturbance/noise studies, other objects | **Not done** (out of scope) |
