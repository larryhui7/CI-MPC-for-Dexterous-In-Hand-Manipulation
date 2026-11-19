# Contact-implicit MPC for Dexterous In-hand Manipulation using DG3F-M for constrained rotation

Reproducing high-level contact-implicit MPC of Jiang et al.,
*Contact-Implicit Model Predictive Control for Dexterous In-hand Manipulation*
(IROS 2024), on a Tesollo DG-3F-M hand rotating a sphere about a fixed vertical
hinge by some angle. We have made some modifications namely not using Croccodyl. 

> MPC actions are executed through the **same smoothed quasi-dynamic model** used for planning.

## Setup and run

```bash
./setup.sh                          # venv + Drake 1.58.0 + build qsim_cpp (~1-2 min)
# RotZ system (paper Fig. 4a): object on a damped vertical hinge
.venv/bin/python run_rotz.py scene       # Step 1: inspect the scene, show it in Meshcat
.venv/bin/python run_rotz.py step        # Step 2: single smoothed contact-model steps
.venv/bin/python run_rotz.py grad        # Step 3: f_x, f_u vs. central finite differences
.venv/bin/python run_rotz.py traj        # Step 4: one DDP solve (N = 10), plot
.venv/bin/python run_rotz.py mpc         # Step 5: closed-loop MPC + baseline + repro check + Meshcat animation

# Free system (paper Fig. 4b): free 6-dof object held only by the fingers (no support), rotate about the vertical axis
.venv/bin/python run_free.py scene       # inspect the Free scene, show it in Meshcat
.venv/bin/python run_free.py traj        # one DDP solve (N = 10)
.venv/bin/python run_free.py mpc         # closed-loop MPC + zero-input baseline + Meshcat animation

# Valve system: a spoked valve on a vertical hinge, turned by pushing its spokes (same commands as run_rotz.py)
.venv/bin/python run_valve.py scene      # inspect the valve scene, show it in Meshcat
.venv/bin/python run_valve.py mpc        # closed-loop MPC + baseline + Meshcat animation

# Same commands on the Allegro right hand (default: --hand dg3fm)
.venv/bin/python run_rotz.py mpc --hand allegro
.venv/bin/python run_free.py mpc --hand allegro
.venv/bin/python run_valve.py mpc --hand allegro
```

You can acess Meshcat at **http://localhost:7123** only while the command is running; press Enter if you want to quit. `--no-meshcat` skips the viewer (and the "press Enter" prompt). All parameters are in the `CONFIGURATION` block at the top of `run_rotz.py`; hand-specific ones (object position, collision spheres, initial grasp) are in `HAND_CFG` right below it, and `run_free.py` overrides only the Free-specific ones. Outputs go to `results/` (Allegro runs are prefixed `allegro_`).

## Files

| File | Purpose |
|---|---|
| `scene.py` | Hand models (`HANDS`: DG-3F-M, Allegro right hand, selected with `--hand`). Generates the planning models from the hand's URDF (sphere collision geometry on the finger links and tips, object SDF, qsim model directive). Contacts in Eq. (2): hand-object and finger-finger, so fingers cannot pass through each other (links of the same finger are filtered). Builds a pydrake plant of the same scene for inspection, grasp IK, signed distances and Meshcat visualization/animation. |
| `ccp.py` | Contact-model interface: equation (2) CCP, smoothed with a log barrier, solved by `qsim_cpp`. Converts the paper's joint-increment input to qsim's position command, returns `f`, `f_x`, `f_u`, and smoothed contact forces. |
| `mpc.py` | Costs (eqs. 3–5) with exact derivatives, control-limited DDP (box-QP backward pass, LM regularization, line search) from Tassal's paper ICRA 2014, joint-limit-aware input bounds, MPC warm-start shift. |
| `run_rotz.py` | RotZ system: configuration block, CLI for Steps 1–5, logging, plots. |
| `run_free.py` | Free system: the object is a free body (q_o = quaternion + position, nx = 19 for the DG-3F-M, 23 for the Allegro hand) held only by the fingers, with no support. Input bound 0.2 rad (paper), W_o diagonal over q_o, object mass 1 kg so the fingers must actually grip it. |
| `run_valve.py` | Valve system: the RotZ setup with a spoked valve (box collision geometry, `object="valve"` in `scene.py`) instead of the sphere, with a fixed valve body under the wheel so fingers cannot reach in underneath; reuses the `run_rotz.py` commands. Uses stiffer contact (`kappa` 1000-3000) than the sphere scripts so the valve only moves while it is being pushed. Valve dimensions and the per-hand starting pose are in its `CONFIGURATION` block; results are prefixed `valve_`. |
| `URDFs/` | Hand descriptions: `dg3fm/` (Tesollo DG-3F-M, 3 fingers x 4 joints) and `allegro/` (Allegro right hand from Drake's `allegro_hand_description`, 4 fingers x 4 joints, mounted palm up; the palm has no collision geometry). |
| `setup.sh`, `requirements.txt` | Dependency build/spec. `setup.sh` pins qsim to commit `c917755` and verifies the Drake tarball's SHA-256. |
| `qsim_drake1.58.patch` | Minimal patch so qsim (written for Drake 1.21) builds and runs on Drake 1.58, plus a fix for qsim crashing when it computes gradients with no contacts. |

Generated (git-ignored): `build/models/*`, `build/free/models/*`, `build/allegro/**`, `build/valve/**` (planning URDF/SDF/directives), `external/` (Drake headers, qsim), `.venv/`.
