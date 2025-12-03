import numpy as np


# ==========================================
# 1. QED (Quadratic Exponential Diffusion)
# ==========================================
def qed_nll_scaled(params, X, dt):
    """
    QED 模型的负对数似然函数
    """
    theta, kappa, g, sigma = params
    if sigma <= 0: return 1e10

    X_curr = X[:-1]
    X_next = X[1:]

    drift = theta * X_curr - kappa * (X_curr ** 2) - g * (X_curr ** 3)
    m_k = X_curr + drift * dt
    v_k = (sigma ** 2) * (X_curr ** 2) * dt

    v_k = np.maximum(v_k, 1e-9)
    nll = np.log(v_k) + (X_next - m_k) ** 2 / v_k
    return 0.5 * np.sum(nll)


# ==========================================
# 2. Hawkes Process
# ==========================================
def calculate_hawkes_intensity(N, mu, alpha, beta, dt):
    """
    计算 Hawkes 过程的强度 lambda
    """
    T = len(N)
    lambdas = np.zeros(T)
    decay = np.exp(-beta * dt)
    D = 0.0
    for t in range(T):
        lambdas[t] = mu + alpha * D
        D = decay * (D + N[t])
    return lambdas


def hawkes_nll_single_series(params, N_arr, dt):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0: return 1e10
    if alpha >= beta: return 1e10

    lambdas = calculate_hawkes_intensity(N_arr, mu, alpha, beta, dt)
    probs = lambdas * dt
    probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
    ll_terms = N_arr * np.log(probs) + (1 - N_arr) * np.log(1 - probs)
    return -np.sum(ll_terms)


def joint_hawkes_nll(params, N_p, N_m, dt):
    """
    联合优化正负跳跃的 Hawkes 参数
    """
    l0_p, l0_m, a_p, a_m, beta = params
    nll_p = hawkes_nll_single_series([l0_p, a_p, beta], N_p, dt)
    nll_m = hawkes_nll_single_series([l0_m, a_m, beta], N_m, dt)
    return nll_p + nll_m


# ==========================================
# 3. Path Simulator
# ==========================================
def simulate_btc_paths(n_paths, n_steps, s0, qed_params, hawkes_params, jump_dist, dt_year, scale_factor=1.0):
    """
    结合 QED 和 Hawkes 的蒙特卡洛模拟 (来自 02 和 04)
    """
    theta, kappa, g, sigma = qed_params
    l0_p, l0_m, a_p, a_m, beta = hawkes_params
    J_plus, J_minus = jump_dist

    x0_scaled = s0 / scale_factor
    y = np.zeros((n_steps, n_paths))
    y[0, :] = np.log(x0_scaled)
    D_p = np.zeros(n_paths)
    D_m = np.zeros(n_paths)
    decay = np.exp(-beta * dt_year)
    sqrt_dt = np.sqrt(dt_year)

    for t in range(n_steps - 1):
        y_curr = y[t, :]
        S_curr = np.exp(y_curr)
        drift = (theta - kappa * S_curr - g * S_curr ** 2) - 0.5 * sigma ** 2
        diff = sigma * np.random.normal(size=n_paths)
        y_cont = y_curr + drift * dt_year + diff * sqrt_dt

        lam_p = l0_p + a_p * D_p
        lam_m = l0_m + a_m * D_m
        p_p = np.clip(lam_p * dt_year, 0, 1)
        p_m = np.clip(lam_m * dt_year, 0, 1)

        N_p = np.random.rand(n_paths) < p_p
        N_m = np.random.rand(n_paths) < p_m
        N_m[N_p & N_m] = False  # 禁止同一时间步同时发生正负跳跃

        J_val_p = np.zeros(n_paths)
        J_val_m = np.zeros(n_paths)
        if np.any(N_p): J_val_p[N_p] = np.random.choice(J_plus, size=np.sum(N_p), replace=True)
        if np.any(N_m): J_val_m[N_m] = np.random.choice(J_minus, size=np.sum(N_m), replace=True)

        y[t + 1, :] = y_cont + J_val_p - J_val_m
        D_p = decay * (D_p + N_p.astype(float))
        D_m = decay * (D_m + N_m.astype(float))

    return np.exp(y) * scale_factor


def generate_candles(close_prices, sigma_daily=0.04):
    """
    从收盘价生成高低点数据
    """
    n_steps, n_paths = close_prices.shape
    highs = np.zeros_like(close_prices)
    lows = np.zeros_like(close_prices)
    step_vol = sigma_daily / np.sqrt(24 * 12)

    for t in range(1, n_steps):
        prev_c = close_prices[t - 1, :]
        curr_c = close_prices[t, :]
        body_max = np.maximum(prev_c, curr_c)
        body_min = np.minimum(prev_c, curr_c)

        # 简单的影线模拟
        u_wick = np.abs(np.random.normal(0, step_vol, n_paths)) * curr_c
        d_wick = np.abs(np.random.normal(0, step_vol, n_paths)) * curr_c

        highs[t, :] = body_max + u_wick
        lows[t, :] = body_min - d_wick

    highs[0, :] = close_prices[0, :]
    lows[0, :] = close_prices[0, :]
    return highs, lows
