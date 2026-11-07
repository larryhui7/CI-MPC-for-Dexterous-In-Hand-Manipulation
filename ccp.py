"""Contact model: smoothed CQDC dynamics (Eq. 2) via qsim_cpp"""
import copy
from types import SimpleNamespace
import numpy as np
import qsim_cpp
import scene


def make_model(cfg, files, kappa=None):
    """Build qsim's quasi-dynamic simulator for the hand + object scene, i.e. the
    dynamics x+ = f(x, u) of CI-MPC Eq. (1) given by the CQDC problem, Eq. (2)."""
    p = qsim_cpp.QuasistaticSimParameters()
    p.h = cfg["h"]  # time step h of Eq. (2) (jiang uses 0.1s)
    p.gravity = np.array([0.0, 0.0, -9.81])  # exerts no torque about the object's z axis
    p.contact_detection_tolerance = cfg["contact_detection_tolerance"]  # pairs farther apart are left out of Eq. (2)
    p.is_quasi_dynamic = True 
    p.unactuated_mass_scale = cfg["Kp"] * cfg["h"] / cfg["hinge_damping"]
    p.log_barrier_weight = cfg["kappa"] if kappa is None else kappa  # kappa (jiang uses 100; but smaller is smoother; did not play yet)
    p.forward_mode = qsim_cpp.ForwardDynamicsMode.kLogIcecream  # analytically smoothed Eq. (2): log barrier on the friction cones
    p.gradient_mode = qsim_cpp.GradientMode.kNone  # set per call in step()
    p.calc_contact_forces = False  # set per call in step()
    p.use_free_solvers = True  # no commercial licence: SCS for the phase-1 SOCP
    p.nd_per_contact = 4       # unused by icecream modes
    # Hand joints are position-controlled with stiffness Kp (stiff springs pulled to
    # q_cmd); the object is the only unactuated body.
    sim = qsim_cpp.QuasistaticSimulatorCpp.make_quasistatic_simulator(
        files["directives"], {"hand": cfg["Kp"] * np.ones(scene.NU)},
        {"object": files["object"]}, p)
    ia, iu = sim.get_q_a_indices_into_q(), sim.get_q_u_indices_into_q()
    n_v_o = scene.n_v_object(cfg)  # 1 (RotZ hinge) or 6 (Free body)
    assert (sim.num_actuated_dofs(), sim.num_unactuated_dofs()) == (scene.NU, n_v_o), \
        f"expected 12 actuated hand joints + {n_v_o} unactuated object dofs"
    perm = np.concatenate([ia, iu])  # x[i] = q_qsim[perm[i]]
    return SimpleNamespace(sim=sim, params=p, perm=perm, h=cfg["h"])


def to_qsim(m, x):
    """Our x -> qsim's q (inverse of perm)."""
    q = np.empty_like(x)
    q[m.perm] = x
    return q


def step(m, x, u, grad=False, contacts=False, kappa=None, exact=False):
    """x+ = f(x, u). Returns x+ or (x+, f_x, f_u); with contacts=True also a
    contact list (see `contact_list`).

    exact=True uses the unsmoothed SOCP (kSocpMp) for comparison only."""
    p = copy.copy(m.params)  # per-call options, the model's defaults stay untouched
    if kappa is not None:
        p.log_barrier_weight = kappa
    if exact:
        p.forward_mode = qsim_cpp.ForwardDynamicsMode.kSocpMp  # Eq. (2) without smoothing
    # kAB: implicit-function derivatives A = dq+/dq, B = dq+/dq_cmd of the smoothed problem
    p.gradient_mode = qsim_cpp.GradientMode.kAB if grad else qsim_cpp.GradientMode.kNone
    p.calc_contact_forces = contacts
    q = to_qsim(m, x)
    # Paper input u = S_r dq (joint increments); qsim takes the absolute command q_cmd = q_r + u.
    q_cmd = np.asarray(x[:scene.NU]) + np.asarray(u)  # paper input -> qsim input
    x_next = m.sim.calc_dynamics(q, q_cmd, p)[m.perm]  # solve Eq. (2), back to x order
    out = [x_next]
    if grad:
        A = m.sim.get_Dq_nextDq()[np.ix_(m.perm, m.perm)]  # dx+/dx with q_cmd held fixed
        B = m.sim.get_Dq_nextDqa_cmd()[m.perm, :]  # dx+/dq_cmd
        # Chain rule through q_cmd = S_r x + u: f_x = A + B S_r, f_u = B
        f_x = A.copy()
        f_x[:, :scene.NU] += B  # d q_cmd / dx = S_r
        out += [f_x, B]
    if contacts:
        out.append(contact_list(m))
    return out[0] if len(out) == 1 else tuple(out)


def contact_list(m):
    """Hand-object contact pairs of the last step (finger-finger pairs left out):
    dicts with the hand body, signed distance phi [m], force on the object f [N]
    (world frame) and contact point p [m]."""
    a, b, phi, f_A, p_WC = m.sim.get_contact_arrays()  # per pair: bodies A, B; phi; force on A; point
    out = []
    for i in range(len(phi)):
        if "object" not in (a[i], b[i]):
            continue
        f_obj = f_A[i] if a[i] == "object" else -f_A[i]  # express as the force on the object
        hand_body = b[i] if a[i] == "object" else a[i]
        out.append(dict(body=hand_body, phi=phi[i], f=f_obj, p=p_WC[i]))
    return out


def per_body(contacts):
    """Merge the contacts of each hand body (a link can carry several spheres):
    body -> dict(phi = closest signed distance [m], f = total force on the object [N])."""
    out = {}
    for c in contacts:
        d = out.setdefault(c["body"], dict(phi=np.inf, f=np.zeros(3)))
        d["phi"], d["f"] = min(d["phi"], c["phi"]), d["f"] + c["f"]
    return out


def object_torque(c, center):
    """Torque of contact c about the object's +z axis through center: (r x f)_z [N m]."""
    r = c["p"] - np.asarray(center)
    return r[0] * c["f"][1] - r[1] * c["f"][0]
