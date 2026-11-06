"""Contact model: the paper's smoothed CQDC dynamics (eq. 2), via qsim_cpp.

Eq. (2), one time step of length h, decision variable v = dq / h:

    min_v  1/2 v'Qv + b'v   s.t.  mu_i ||J_it v|| <= J_in v + phi_i / h   (CCP/SOCP)

  Q = blkdiag(Kp h^2 (hand joints), Q_u (object)),
  b = -h [Kp (q_cmd - q_r); tau_ext_object],

i.e. each hand joint is a stiff spring pulled to its position command q_cmd,
and the object has no momentum (quasi-dynamic). qsim's ForwardDynamicsMode
kLogIcecream solves the analytically smoothed version (paper Sec. III-A.2):

    min_v  kappa (1/2 v'Qv + b'v) - sum_i log(w_i0^2 - ||w_it||^2),
    w_i = J_i v + [phi_i / (h mu_i), 0, 0],   J_i = [J_in / mu_i; J_it],

with kappa = log_barrier_weight (paper: kappa = 100; smaller = smoother).
The contact impulses are the barrier's implicit dual (computed in our qsim
patch): lambda_i = 2 / (kappa (w_i0^2 - ||w_it||^2)) [w_i0, -w_i1, -w_i2].

Input convention. qsim's input is the ABSOLUTE joint position command q_cmd.
The paper uses joint increments u = S_r dq. We set q_cmd = q_r + u = S_r x + u:

    f(x, u) = qsim(x, S_r x + u),   f_x = A + B S_r,   f_u = B,

where A = dq+/dq (q_cmd fixed) and B = dq+/dq_cmd come from qsim (GradientMode
kAB, the implicit-function derivatives of the smoothed problem).

Object hinge. With no momentum, the object's Q_u acts as a viscous damper:
impulse h*tau = Q_u * omega  =>  Q_u = h * b_hinge (paper Fig. 4a: damped hinge).
qsim sets Q_u = min(Kp) h^2 / unactuated_mass_scale, so we pick
unactuated_mass_scale = Kp h / b_hinge.
"""
import copy
from types import SimpleNamespace

import numpy as np
import qsim_cpp

import scene


def make_model(cfg, files, kappa=None):
    p = qsim_cpp.QuasistaticSimParameters()
    p.h = cfg["h"]
    p.gravity = np.array([0.0, 0.0, -9.81])  # exerts no torque about the object's z axis
    p.contact_detection_tolerance = cfg["contact_detection_tolerance"]
    p.is_quasi_dynamic = True
    p.unactuated_mass_scale = cfg["Kp"] * cfg["h"] / cfg["hinge_damping"]
    p.log_barrier_weight = cfg["kappa"] if kappa is None else kappa
    p.forward_mode = qsim_cpp.ForwardDynamicsMode.kLogIcecream
    p.gradient_mode = qsim_cpp.GradientMode.kNone
    p.calc_contact_forces = False
    p.use_free_solvers = True  # no commercial licence: SCS for the phase-1 SOCP
    p.nd_per_contact = 4       # unused by icecream modes
    sim = qsim_cpp.QuasistaticSimulatorCpp.make_quasistatic_simulator(
        files["directives"], {"hand": cfg["Kp"] * np.ones(scene.NU)},
        {"object": files["object"]}, p)
    ia, iu = sim.get_q_a_indices_into_q(), sim.get_q_u_indices_into_q()
    assert (sim.num_actuated_dofs(), sim.num_unactuated_dofs()) == (scene.NU, 1), \
        "expected 12 actuated hand joints + 1 unactuated object joint"
    perm = np.concatenate([ia, iu])  # x[i] = q_qsim[perm[i]]
    return SimpleNamespace(sim=sim, params=p, perm=perm, h=cfg["h"])


def to_qsim(m, x):
    q = np.empty_like(x)
    q[m.perm] = x
    return q


def step(m, x, u, grad=False, contacts=False, kappa=None, exact=False):
    """x+ = f(x, u). Returns x+ or (x+, f_x, f_u); with contacts=True also a
    contact list (see `contact_list`).

    exact=True uses the unsmoothed SOCP (kSocpMp) for comparison only."""
    p = copy.copy(m.params)
    if kappa is not None:
        p.log_barrier_weight = kappa
    if exact:
        p.forward_mode = qsim_cpp.ForwardDynamicsMode.kSocpMp
    p.gradient_mode = qsim_cpp.GradientMode.kAB if grad else qsim_cpp.GradientMode.kNone
    p.calc_contact_forces = contacts
    q = to_qsim(m, x)
    q_cmd = np.asarray(x[:scene.NU]) + np.asarray(u)  # paper input -> qsim input
    x_next = m.sim.calc_dynamics(q, q_cmd, p)[m.perm]
    out = [x_next]
    if grad:
        A = m.sim.get_Dq_nextDq()[np.ix_(m.perm, m.perm)]
        B = m.sim.get_Dq_nextDqa_cmd()[m.perm, :]
        fx = A.copy()
        fx[:, :scene.NU] += B  # d q_cmd / dx = S_r
        out += [fx, B]
    if contacts:
        out.append(contact_list(m))
    return out[0] if len(out) == 1 else tuple(out)


def contact_list(m):
    """Contact pairs of the last step: dicts with the hand body, signed distance
    phi [m], force on the object f [N] (world frame), point p [m], and the
    torque about the object's +z axis tau_z [N m]."""
    a, b, phi, f_A, p_WC = m.sim.get_contact_arrays()
    out = []
    for i in range(len(phi)):
        f_obj = f_A[i] if a[i] == "object" else -f_A[i]
        hand_body = b[i] if a[i] == "object" else a[i]
        out.append(dict(body=hand_body, phi=phi[i], f=f_obj, p=p_WC[i]))
    return out


def object_torque(c, center):
    r = c["p"] - np.asarray(center)
    return r[0] * c["f"][1] - r[1] * c["f"][0]
