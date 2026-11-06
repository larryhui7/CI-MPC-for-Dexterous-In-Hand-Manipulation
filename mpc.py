"""High-level contact-implicit MPC: costs (paper eqs. 3-5) and control-limited DDP.

Optimal control problem (paper eq. 1):
    min_U  sum_{i<N} l(x_i, u_i) + l_f(x_N)   s.t.  x_{i+1} = f(x_i, u_i),
           u_lo <= u_i <= u_hi
    l   = C_o(x) + C_r(x) + C_u(u),   l_f = C_o(x_N) + C_r(x_N)   (terminal C_r
          optionally scaled up, paper: "increase W_r near the end of the horizon")
    C_o = w_o * e_theta^2,  e_theta = wrap(theta - theta_ref)   (local angle error)
    C_r = ||q_r - q_r,ref||^2_{W_r}   (nominal hand posture = initial grasp)
    C_u = ||u||^2_{W_u}

Solver: control-limited DDP (Tassa, Mansard & Todorov, ICRA 2014) with the
second-order dynamics terms f_xx, f_xu, f_uu omitted (as in the paper), i.e.
iLQR with a box-constrained QP for the control step in every backward-pass
stage, Levenberg-Marquardt regularization of V_xx, and a backtracking line
search on the closed-loop forward rollout.

Joint limits (implementation addition; the paper only states input bounds):
with q_cmd = q_r + u, the bound on u at a stage with hand position q_r is
    max(-u_max, q_lo - q_r) <= u <= min(u_max, q_hi - q_r),
i.e. joint-position commands never leave the joint limits. The backward pass
uses the bounds at the nominal state; the forward pass re-evaluates them at
the actual state and clamps (part of the Tassa forward pass).
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import ccp
import scene

NR = scene.NU  # hand joints = first NR entries of x


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_costs(cfg, q_nominal, theta_ref):
    """Quadratic stage/terminal costs and their exact derivatives."""
    w_o, w_r, w_u = cfg["w_obj"], cfg["w_posture"], cfg["w_u"]
    w_rf = w_r * cfg["w_posture_terminal_scale"]

    def stage(x, u, terminal=False):
        wr = w_rf if terminal else w_r
        e_o = wrap(x[NR] - theta_ref)
        e_r = x[:NR] - q_nominal
        c = w_o * e_o ** 2 + wr * e_r @ e_r
        lx = np.concatenate([2 * wr * e_r, [2 * w_o * e_o]])
        lxx = np.diag(np.concatenate([2 * wr * np.ones(NR), [2 * w_o]]))
        if terminal:
            return c, lx, lxx
        c += w_u * u @ u
        return c, lx, lxx, 2 * w_u * u, 2 * w_u * np.eye(len(u))

    return stage


def input_bounds(cfg, x, q_lo, q_hi):
    lo = np.maximum(-cfg["u_max"], q_lo - x[:NR])
    hi = np.minimum(cfg["u_max"], q_hi - x[:NR])
    return np.minimum(lo, hi), hi


def rollout(m, x0, U, cost, cfg=None, q_lim=None, X_ref=None, K=None):
    """Simulate x_{i+1} = f(x_i, u_i). With feedback (X_ref, K) and bounds this
    is the DDP forward pass; returns X, the (possibly clamped) U, total cost."""
    N = len(U)
    X = np.zeros((N + 1, len(x0)))
    X[0] = x0
    U = U.copy()
    total = 0.0
    for i in range(N):
        if K is not None:
            U[i] = U[i] + K[i] @ (X[i] - X_ref[i])
        if q_lim is not None:
            U[i] = np.clip(U[i], *input_bounds(cfg, X[i], *q_lim))
        X[i + 1] = ccp.step(m, X[i], U[i])
        total += cost(X[i], U[i])[0]
    total += cost(X[N], None, terminal=True)[0]
    return X, U, total


def box_qp(H, g, lo, hi, x0, max_iter=100, min_grad=1e-10, armijo=0.1):
    """Projected-Newton solver for min 1/2 x'Hx + g'x s.t. lo <= x <= hi
    (after Tassa et al. 2014, boxQP.m). Returns (x, free mask, Cholesky factor
    of H[free, free]) or None if H[free, free] is not positive definite."""
    x = np.clip(x0, lo, hi)
    free = np.ones(len(x), bool)
    chol = None
    value = x @ g + 0.5 * x @ H @ x
    for _ in range(max_iter):
        grad = g + H @ x
        clamped = ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0))
        new_free = ~clamped
        if not new_free.any():
            return x, new_free, None
        if chol is None or (new_free != free).any():
            try:
                chol = cho_factor(H[np.ix_(new_free, new_free)])
            except np.linalg.LinAlgError:
                return None
        free = new_free
        if np.linalg.norm(grad[free]) < min_grad:
            break
        # Newton step on the free set, clamped variables held fixed.
        g_free = g[free] + H[np.ix_(free, ~free)] @ x[~free]
        search = np.zeros_like(x)
        search[free] = -cho_solve(chol, g_free) - x[free]
        sdotg = search @ grad
        if sdotg >= -1e-16:
            break
        step = 1.0
        while True:
            xc = np.clip(x + step * search, lo, hi)
            vc = xc @ g + 0.5 * xc @ H @ xc
            if (vc - value) / (step * sdotg) >= armijo:
                break
            step *= 0.6
            if step < 1e-20:
                return x, free, chol
        x, value = xc, vc
    return x, free, chol


def backward_pass(fx, fu, X, U, cost, cfg, q_lim, mu):
    N, nx, nu = len(U), X.shape[1], U.shape[1]
    _, Vx, Vxx = cost(X[N], None, terminal=True)
    k_ff, K = np.zeros((N, nu)), np.zeros((N, nu, nx))
    dV = np.zeros(2)
    for i in reversed(range(N)):
        _, lx, lxx, lu, luu = cost(X[i], U[i])
        Qx = lx + fx[i].T @ Vx
        Qu = lu + fu[i].T @ Vx
        Qxx = lxx + fx[i].T @ Vxx @ fx[i]
        Vreg = Vxx + mu * np.eye(nx)          # Tassa: regularize V_xx
        Quu = luu + fu[i].T @ Vreg @ fu[i]
        Qux = fu[i].T @ Vreg @ fx[i]
        lo, hi = input_bounds(cfg, X[i], *q_lim)
        res = box_qp(Quu, Qu, lo - U[i], hi - U[i],
                     k_ff[i + 1] if i + 1 < N else np.zeros(nu))
        if res is None:
            return None
        k_ff[i], free, chol = res
        if chol is not None:
            K[i][free] = -cho_solve(chol, Qux[free])
        dV += [k_ff[i] @ Qu, 0.5 * k_ff[i] @ Quu @ k_ff[i]]
        Vx = Qx + K[i].T @ Quu @ k_ff[i] + K[i].T @ Qu + Qux.T @ k_ff[i]
        Vxx = Qxx + K[i].T @ Quu @ K[i] + K[i].T @ Qux + Qux.T @ K[i]
        Vxx = 0.5 * (Vxx + Vxx.T)
    return k_ff, K, dV


def ilqr(m, x0, U0, cost, cfg, q_lim, max_iter):
    """Control-limited iLQR. Returns (X, U, info) where info holds the cost
    history, iteration count and the reason for termination."""
    mu, mu_min, mu_max, mu_factor = cfg["mu_init"], 1e-6, 1e10, 2.0
    alphas = 10.0 ** np.linspace(0, -3, 11)
    X, U, J = rollout(m, x0, U0, cost, cfg, q_lim)  # clamp the initial guess
    costs, reason, n_iter = [J], "max_iter", 0
    for n_iter in range(1, max_iter + 1):
        fx, fu = [], []
        for i in range(len(U)):
            _, A, B = ccp.step(m, X[i], U[i], grad=True)
            fx.append(A)
            fu.append(B)
        while True:  # backward pass, increase regularization until it succeeds
            bp = backward_pass(fx, fu, X, U, cost, cfg, q_lim, mu)
            if bp is not None:
                break
            mu = max(mu * mu_factor, mu_min) * 10
            if mu > mu_max:
                return X, U, dict(costs=costs, iters=n_iter, reason="mu_max (backward)")
        k_ff, K, dV = bp
        if np.max(np.abs(k_ff)) < cfg["tol_step"]:
            reason = "small step"
            break
        accepted = False
        for alpha in alphas:  # forward pass with line search
            try:
                Xn, Un, Jn = rollout(m, x0, U + alpha * k_ff, cost, cfg, q_lim, X, K)
            except RuntimeError:  # qsim solver failure -> reject this step size
                continue
            expected = -alpha * (dV[0] + alpha * dV[1])
            if expected > 0 and (J - Jn) / expected > 1e-4:
                accepted = True
                break
        if not accepted:
            mu = max(mu * mu_factor, mu_min) * 10
            if mu > mu_max:
                reason = "mu_max (line search)"
                break
            costs.append(J)
            continue
        mu = mu / mu_factor if mu / mu_factor > mu_min else 0.0
        rel = (J - Jn) / max(abs(J), 1e-12)
        X, U, J = Xn, Un, Jn
        costs.append(J)
        if rel < cfg["tol_rel_cost"]:
            reason = "small cost decrease"
            break
    return X, U, dict(costs=costs, iters=n_iter, reason=reason)


def mpc_solve(m, x, U_warm, cost, cfg, q_lim, max_iter):
    t = time.perf_counter()
    X, U, info = ilqr(m, x, U_warm, cost, cfg, q_lim, max_iter)
    info["time"] = time.perf_counter() - t
    return X, U, info


def shift(U):
    """Warm start for the next MPC step: drop u_0, repeat the last input."""
    return np.vstack([U[1:], U[-1:]])
