"""Valve system: a spoked valve on a damped vertical hinge, turned by the
fingers pushing on its spokes. Same state (x = [hand joints; valve angle]),
costs (Eqs. 3-5), dynamics (Eq. 2), solver and commands as run_rotz.py; only
the object and the initial hand pose differ (planner verification only).

  .venv/bin/python run_valve.py scene    load/inspect the valve scene, show it in Meshcat
  .venv/bin/python run_valve.py step     single smoothed contact-model steps
  .venv/bin/python run_valve.py grad     f_x, f_u vs. finite differences
  .venv/bin/python run_valve.py traj     one DDP trajectory optimization (N=10)
  .venv/bin/python run_valve.py mpc      closed-loop MPC (+ zero-input baseline, Meshcat)

--hand dg3fm (default) or --hand allegro selects the hand model (scene.HANDS).
"""
import argparse
import os

import numpy as np

import run_rotz as rz

# ============================ CONFIGURATION ==================================
# Everything not listed here is shared with run_rotz.CFG / run_rotz.HAND_CFG.
VALVE_CFG = dict(
    object="valve",                   # spoked valve instead of the sphere (scene.py)
    valve_spokes=3,                   # boxes radiating from the hinge axis
    valve_spoke_length=0.05,          # [m] hinge axis to spoke end
    valve_spoke_width=0.016,          # [m]
    valve_height=0.03,                # [m] along the hinge axis
    kappa=1000.0,
    omega_ref=0.79,                   # [rad/s] commanded valve rate
    mpc_steps=80,                     # closed-loop steps (8 s)
)
VALVE_HAND_CFG = {
    "dg3fm": dict(
        build_dir="build/valve",
        result_prefix="valve_",
        grasp_radius=0.035,
        valve_spoke_offset=np.radians(60.0),
        valve_base_radius=0.025,
        valve_base_bottom=0.16,       # [m] clear of the proximal finger links
    ),
    "allegro": dict(
        build_dir="build/allegro/valve",
        result_prefix="valve_allegro_",
        # centered over the palm (palm_link spans x = -0.095..0, y = +-0.0565)
        obj_center=[-0.0475, 0.0, 0.09],
        # a wide, flat valve gripped near the spoke ends: index, middle, ring
        # reach it from -x without curling right back, the thumb from +y
        valve_spoke_length=0.065,
        valve_height=0.02,
        grasp_radius=0.063,
        valve_spoke_offset=np.radians(0.0),
        kappa=3000.0,                 # creep with the hand held still: 1.5 deg in 3 s
        valve_base_radius=0.035,
        valve_base_bottom=0.011,      # [m] palm surface
        # half the DG-3F-M rate: at 0.79 rad/s this hand falls behind and stalls
        # near 190 deg once it can no longer reach under the wheel
        omega_ref=0.4,
        mpc_steps=160,                # 16 s, ~367 deg
        finger_azimuth=np.radians([140.0, 180.0, -140.0, 100.0]),
    ),
}
# =============================================================================


def select_hand(name):
    """Switch run_rotz (and scene.py) to hand `name` with the valve as object."""
    cfg = rz.select_hand(name)
    cfg.update(VALVE_CFG)
    cfg.update(VALVE_HAND_CFG[name])
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scene", "step", "grad", "traj", "mpc"])
    ap.add_argument("--hand", choices=list(VALVE_HAND_CFG), default="dg3fm", help="hand model")
    ap.add_argument("--no-meshcat", action="store_true", help="skip Meshcat")
    ap.add_argument("--plant", choices=["same", "exact"], default="same",
                    help="mpc: execute actions in the planning model (default) or, as a "
                         "diagnostic, in the exact unsmoothed SOCP")
    args = ap.parse_args()
    cfg = select_hand(args.hand)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    {"scene": rz.cmd_scene, "step": rz.cmd_step, "grad": rz.cmd_grad,
     "traj": rz.cmd_traj, "mpc": rz.cmd_mpc}[args.command](args)


if __name__ == "__main__":
    main()
