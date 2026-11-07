"""High-level CI-MPC and control-limited DDP
with joint limits (Jiang et. al did not have joint limits)
"""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import ccp
import scene

NR = scene.NU  # number of joints in the hand = first NR entries of x


def make_costs(cfg, q_r_ref, q_o_ref):
    """q_o_ref: object reference per stage, shape (N + 1, n_o) or (N + 1,) for a
    scalar q_o, i.e. the reference trajectory q_o,ref(t_i) over the horizon
    (index N = terminal). q_o = x[NR:]: yaw (RotZ) or quaternion + position (Free).
    W_o, W_r may be scalars or per-coordinate diagonals."""
    q_o_ref = np.asarray(q_o_ref, float)
    q_o_ref = q_o_ref.reshape(len(q_o_ref), -1)
    # weight matrices
    W_o, W_r, W_u = cfg["w_obj"], cfg["w_posture"], cfg["w_u"]
    W_r_f = W_r * cfg["w_posture_terminal_scale"]  # terminal weight

    def stage(x, u, i, terminal=False):
        W_r_i = W_r_f if terminal else W_r
        q_r, q_o = x[:NR], x[NR:]
        e_o = q_o - q_o_ref[i]
        e_r = q_r - q_r_ref
        C_o = e_o @ (W_o * e_o)  # Eq.(4)
        C_r = W_r_i * e_r @ e_r  # Eq.(4)
        # deriv. of l(x,u) w.r.t. to x
        l_x = np.concatenate([2 * W_r_i * e_r, 2 * W_o * e_o])
        # 2nd deriv. of l(x,u) w.r.t. to x
        l_xx = np.diag(np.concatenate([2 * W_r_i * np.ones(NR), 2 * W_o * np.ones(len(q_o))]))
        if terminal:
            l_f = C_o + C_r  # Eq.(3)
            return l_f, l_x, l_xx
        C_u = W_u * u @ u  # Eq.(5)
        l = C_o + C_r + C_u  # Eq.(3)
        l_u = 2 * W_u * u  # deriv. of l(x,u) w.r.t. to u
        l_uu = 2 * W_u * np.eye(len(u))  # 2nd deriv. of l(x,u) w.r.t. to u
        return l, l_x, l_xx, l_u, l_uu

    return stage


def input_bounds(cfg, x, q_lo, q_hi):
    q_r = x[:NR]
    min_u = np.maximum(-cfg["u_max"], q_lo - q_r)
    max_u = np.minimum(cfg["u_max"], q_hi - q_r)
    return np.minimum(min_u, max_u), max_u


def rollout(m, x0, U, cost, cfg=None, q_lim=None, X_ref=None, K=None,
            k_ff=None, alpha=1.0):
    N = len(U)
    X = np.zeros((N + 1, len(x0)))
    X[0] = x0  # Tassa et. al. Eq.(7a)
    U = U.copy()
    total = 0.0  # Jiang et. al. Eq.(1) cost
    for i in range(N):
        if K is not None:
            # Tassa et. al. Eq.(7b)
            U[i] = U[i] + alpha * k_ff[i] + K[i] @ (X[i] - X_ref[i])
        if q_lim is not None:
            # Tassa Eq. (9) ONLY for feasibility; K and k come from box QP.
            U[i] = np.clip(U[i], *input_bounds(cfg, X[i], *q_lim))
        X[i + 1] = ccp.step(m, X[i], U[i])  # Tassa et. al. Eq.(7c)
        total += cost(X[i], U[i], i)[0]  # Jiang et. al. Eq.(3) stage cost
    total += cost(X[N], None, N, terminal=True)[0]  # terminal cost
    return X, U, total


def box_qp(H, g, min_du, max_du, du0, max_iter=100, min_grad=1e-10, armijo=0.1):
    """Projected-Newton box QP, Tassa et. al. Appendix I"""
    du = np.clip(du0, min_du, max_du)
    free = np.ones(len(du), bool)
    chol = None
    value = du @ g + 0.5 * du @ H @ du  # Tassa Eq. (14a)
    for _ in range(max_iter):  # Tassa Alg. 1
        grad = g + H @ du
        clamped = ((du <= min_du) & (grad > 0)) | ((du >= max_du)
                                                   & (grad < 0))  # Tassa Alg. 1 step 1, Eq. (15)
        new_free = ~clamped
        if not new_free.any():
            return du, new_free, None
        if chol is None or (new_free != free).any():
            try:
                # Tassa Eq. (16)
                chol = cho_factor(H[np.ix_(new_free, new_free)])
            except np.linalg.LinAlgError:
                return None
        free = new_free
        if np.linalg.norm(grad[free]) < min_grad:  # Tassa Alg. 1 step 3
            break
        # Tassa Alg. 1 step 2, Eqs. (16)-(17)
        g_free = g[free] + H[np.ix_(free, ~free)] @ du[~free]
        search = np.zeros_like(du)
        search[free] = -cho_solve(chol, g_free) - du[free]
        sdotg = search @ grad
        if sdotg >= -1e-16:
            break
        step = 1.0  # Tassa Alg. 1 step 4
        while True:
            du_c = np.clip(du + step * search, min_du,
                           max_du)  # Tassa Eq. (18)
            value_c = du_c @ g + 0.5 * du_c @ H @ du_c  # Tassa Eq. (14a)
            if (value_c - value) / (step * sdotg) >= armijo:  # Tassa Eq. (19)
                break
            step *= 0.6
            if step < 1e-20:
                return du, free, chol
        du, value = du_c, value_c
    return du, free, chol


def backward_pass(f_x, f_u, X, U, cost, cfg, q_lim, mu):
    """DDP backward pass, Tassa et. al. Sec. II-B and III-C (f_xx, f_ux, f_uu
    terms of Eq. (4) omitted, as in the CI-MPC paper)"""
    N, nx, nu = len(U), X.shape[1], U.shape[1]
    _, V_x, V_xx = cost(X[N], None, N, terminal=True)  # V_N = l_f(x_N)
    k_ff, K = np.zeros((N, nu)), np.zeros((N, nu, nx))
    # expected cost change [k'Q_u, 1/2 k'Q_uu k], for the line search
    dV = np.zeros(2)
    for i in reversed(range(N)):
        _, l_x, l_xx, l_u, l_uu = cost(X[i], U[i], i)
        Q_x = l_x + f_x[i].T @ V_x  # Eq. (4a)
        Q_u = l_u + f_u[i].T @ V_x  # Eq. (4b)
        Q_xx = l_xx + f_x[i].T @ V_xx @ f_x[i]  # Eq. (4c)
        Q_uu = l_uu + f_u[i].T @ V_xx @ f_u[i]  # Eq. (4e)
        Q_ux = f_u[i].T @ V_xx @ f_x[i]  # Eq. (4d), l_ux = 0
        # Sec. II-D, regularization (on V_xx, Tassa IROS 2012): the regularized
        # Q_uu, Q_ux are used only for the gains k, K; dV and the V update below
        # use the unregularized ones
        V_xx_reg = V_xx + mu * np.eye(nx)
        Q_uu_reg = l_uu + f_u[i].T @ V_xx_reg @ f_u[i]
        Q_ux_reg = f_u[i].T @ V_xx_reg @ f_x[i]
        min_u, max_u = input_bounds(cfg, X[i], *q_lim)
        # Eq. (11): k = argmin Q over du s.t. the box, warm-started from k_{i+1}
        res = box_qp(Q_uu_reg, Q_u, min_u - U[i], max_u - U[i],
                     k_ff[i + 1] if i + 1 < N else np.zeros(nu))
        if res is None:
            return None
        k_ff[i], free, chol = res
        if chol is not None:
            # Sec. III-C: K_f = -Q_uu,f^-1 Q_ux, K_c = 0
            K[i][free] = -cho_solve(chol, Q_ux_reg[free])
        dV += [k_ff[i] @ Q_u, 0.5 * k_ff[i] @ Q_uu @
               k_ff[i]]  # Eq. (6a), general form
        # Eqs. (6b)-(6c), in the general form valid for any k, K (the box-QP
        # k is not -Q_uu^-1 Q_u, so the simplified (6b)-(6c) do not apply)
        V_x = Q_x + K[i].T @ Q_uu @ k_ff[i] + K[i].T @ Q_u + Q_ux.T @ k_ff[i]
        V_xx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]
        V_xx = 0.5 * (V_xx + V_xx.T)  # symmetric part of V_xx
    return k_ff, K, dV


def box_ddp(m, x0, U0, cost, cfg, q_lim, max_iter):
    """Control-limited DDP [Tassa 2014] with f_xx, f_xu, f_uu omitted (CI-MPC
    "Generation of Contact References"). Solves CI-MPC Eq. (1). Returns (X, U, info) where info holds the cost history, iteration count and
    the reason for termination. basically iLQR"""
    # Tassa Sec. II-D
    mu, mu_min, mu_max, mu_factor = cfg["mu_init"], 1e-6, 1e10, 2.0
    alphas = 10.0 ** np.linspace(0, -3, 11)  # line-search alphas, Tassa Sec. II-C: 1, then reduced
    X, U, J = rollout(m, x0, U0, cost, cfg, q_lim)  # nominal trajectory, Tassa Eq. (1)
    costs, reason, n_iter = [J], "max_iter", 0
    for n_iter in range(1, max_iter + 1):
        # Linearize f along the nominal trajectory (f_x, f_u in Tassa Eq. (4))
        f_x, f_u = [], []
        for i in range(len(U)):
            _, A, B = ccp.step(m, X[i], U[i], grad=True)
            f_x.append(A)
            f_u.append(B)
        while True:  # backward pass (Tassa Sec. II-B, III-C); raise mu until it succeeds
            bp = backward_pass(f_x, f_u, X, U, cost, cfg, q_lim, mu)
            if bp is not None:
                break
            mu = max(mu * mu_factor, mu_min) * 10
            if mu > mu_max:
                return X, U, dict(costs=costs, iters=n_iter, reason="mu_max (backward)")
        k_ff, K, dV = bp
        if np.max(np.abs(k_ff)) < cfg["tol_step"]:  # converged
            reason = "small step"
            break
        accepted = False
        for alpha in alphas:  # forward pass with line search, Tassa Eq. (7)
            try:
                Xn, Un, Jn = rollout(m, x0, U, cost, cfg, q_lim,
                                     X_ref=X, K=K, k_ff=k_ff, alpha=alpha)
            except RuntimeError:  # qsim solver failure -> reject this step size
                continue
            # Expected reduction from Tassa Eq. (6a), general form:
            expected = -alpha * (dV[0] + alpha * dV[1])
            if expected > 0 and (J - Jn) / expected > 1e-4:
                accepted = True
                break
        if not accepted:  # no alpha worked: increase mu, retry from the same trajectory
            mu = max(mu * mu_factor, mu_min) * 10
            if mu > mu_max:
                reason = "mu_max (line search)"
                break
            costs.append(J)
            continue
        mu = mu / mu_factor if mu / mu_factor > mu_min else 0.0  # step accepted: decrease mu
        rel = (J - Jn) / max(abs(J), 1e-12)
        X, U, J = Xn, Un, Jn  # new nominal trajectory
        costs.append(J)
        if rel < cfg["tol_rel_cost"]:  # converged: relative cost decrease small
            reason = "small cost decrease"
            break
    return X, U, dict(costs=costs, iters=n_iter, reason=reason)


def mpc_solve(m, x, U_warm, cost, cfg, q_lim, max_iter):
    t = time.perf_counter()
    X, U, info = box_ddp(m, x, U_warm, cost, cfg, q_lim, max_iter)
    info["time"] = time.perf_counter() - t
    return X, U, info


def shift(U):
    """Warm start for the next MPC step: drop u_0, repeat the last input."""
    return np.vstack([U[1:], U[-1:]])
