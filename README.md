# Contact-implicit MPC for Dexterous In-hand Manipulation using DG3F-M for constrained rotation

Reproducing **high-level contact-implicit MPC** of Jiang et al.,
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
```

You can acess Meshcat at **http://localhost:7123** only while the command is running; press Enter if you want to quit. `--no-meshcat` skips the viewer (and the "press Enter" prompt). All parameters are in the `CONFIGURATION` block at the top of `run_rotz.py`; `run_free.py` overrides only the Free-specific ones. Outputs go to `results/`.

## Files

| File | Purpose |
|---|---|
| `scene.py` | Generates the planning models from the official URDF (sphere collision geometry covering finger links 2, 3, 4 and the tip, object SDF, qsim model directive). Contacts in Eq. (2): hand-object and finger-finger, so fingers cannot pass through each other (links of the same finger are filtered). Builds a pydrake plant of the same scene for inspection, grasp IK, signed distances and Meshcat visualization/animation. |
| `ccp.py` | Contact-model interface: equation (2) CCP, smoothed with a log barrier, solved by `qsim_cpp`. Converts the paper's joint-increment input to qsim's position command, returns `f`, `f_x`, `f_u`, and smoothed contact forces. |
| `mpc.py` | Costs (eqs. 3–5) with exact derivatives, control-limited DDP (box-QP backward pass, LM regularization, line search) from Tassal's paper ICRA 2014, joint-limit-aware input bounds, MPC warm-start shift. |
| `run_rotz.py` | RotZ system: configuration block, CLI for Steps 1–5, logging, plots. |
| `run_free.py` | Free system: the object is a free body (q_o = quaternion + position, nx = 19) held only by the three fingers, with no support. Input bound 0.2 rad (paper), W_o diagonal over q_o, object mass 1 kg so the fingers must actually grip it. |
| `setup.sh`, `requirements.txt` | Dependency build/spec. `setup.sh` pins qsim to commit `c917755` and verifies the Drake tarball's SHA-256. |
| `qsim_drake1.58.patch` | Minimal patch so qsim (written for Drake 1.21) builds and runs on Drake 1.58, plus a fix for qsim crashing when it computes gradients with no contacts. |

Generated (git-ignored): `build/models/*`, `build/free/models/*` (planning URDF/SDF/directives), `external/` (Drake headers, qsim), `.venv/`.