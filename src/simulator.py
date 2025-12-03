import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from typing import Tuple, Optional, List

class BTCPriceSimulator:
    """BTC价格模拟器：结合QED扩散过程（连续波动）和Hawkes跳跃过程（事件驱动跳变）"""
    
    def __init__(self, 
                 qed_params: dict, 
                 hawkes_params: dict, 
                 s0: float = 50000.0, 
                 seed: int = 42):
        """
        初始化参数
        :param qed_params: QED扩散模型参数 {'theta': 净增长率, 'kappa': 均值回归强度, 'g': 非线性饱和项, 'sigma': 波动率系数, 'omega': 额外冲击系数}
        :param hawkes_params: Hawkes跳跃模型参数 {'mu': 背景强度, 'alpha': 激发系数, 'beta': 衰减系数, 'jump_mean': 跳跃均值, 'jump_std': 跳跃标准差}
        :param s0: 初始价格
        :param seed: 随机种子
        """
        self.s0 = s0
        self.qed_params = self._validate_qed_params(qed_params)
        self.hawkes_params = self._validate_hawkes_params(hawkes_params)
        self.rng = np.random.default_rng(seed)
        
        # Hawkes过程状态
        self.current_intensity = self.hawkes_params['mu']  # 当前跳跃强度
    
    def _validate_qed_params(self, params: dict) -> dict:
        """验证QED参数完整性"""
        required = ['theta', 'kappa', 'g', 'sigma', 'omega']
        if not all(k in params for k in required):
            raise ValueError(f"QED参数缺失，需包含：{required}")
        return params
    
    def _validate_hawkes_params(self, params: dict) -> dict:
        """验证Hawkes参数完整性"""
        required = ['mu', 'alpha', 'beta', 'jump_mean', 'jump_std']
        if not all(k in params for k in required):
            raise ValueError(f"Hawkes参数缺失，需包含：{required}")
        if params['alpha'] >= params['beta']:
            raise ValueError("Hawkes过程需满足 alpha < beta（保证强度平稳）")
        return params
    
    def _qed_diffusion_step(self, x_t: float, dt: float) -> float:
        """QED扩散过程单步更新（连续波动部分）"""
        theta = self.qed_params['theta']
        kappa = self.qed_params['kappa']
        g = self.qed_params['g']
        sigma = self.qed_params['sigma']
        omega = self.qed_params['omega']
        
        # 漂移项：非线性均值回归
        drift = kappa * x_t * (theta / kappa - x_t - (g / kappa) * x_t**2) * dt
        # 扩散项：乘法波动率 + 额外冲击
        shock = self.rng.normal(0, np.sqrt(dt))
        diffusion = sigma * x_t * (shock + omega * self.rng.normal(0, 1) * np.sqrt(dt))
        return x_t + drift + diffusion
    
    def _hawkes_jump_step(self, dt: float) -> Tuple[bool, float]:
        """Hawkes跳跃过程单步更新（事件驱动跳变）"""
        mu = self.hawkes_params['mu']
        alpha = self.hawkes_params['alpha']
        beta = self.hawkes_params['beta']
        
        # 1. 更新跳跃强度（自激发过程）
        self.current_intensity = mu + (self.current_intensity - mu) * np.exp(-beta * dt)
        
        # 2. 生成跳跃事件（泊松过程）
        jump_count = self.rng.poisson(self.current_intensity * dt)
        has_jump = jump_count > 0
        
        # 3. 生成跳跃幅度（对数收益率）
        if has_jump:
            jump_size = self.rng.normal(self.hawkes_params['jump_mean'], self.hawkes_params['jump_std'])
            # 跳跃后强度激发
            self.current_intensity += alpha * jump_count
            return True, jump_size
        return False, 0.0
    
    def simulate_path(self, 
                     T: float = 7.0,  # 总时间（天）
                     dt: float = 1/24/12  # 时间步长（5分钟）
                     ) -> pd.DataFrame:
        """
        生成完整价格路径
        :param T: 模拟总时长（天）
        :param dt: 时间步长（天），默认5分钟
        :return: 包含时间、价格、收益率、波动率、跳跃标记的DataFrame
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps)
        prices = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        vol_15m = np.zeros(n_steps)  # 15分钟滚动波动率
        has_jump = np.zeros(n_steps, dtype=bool)
        
        # 初始化
        prices[0] = self.s0
        returns[0] = 0.0
        self.current_intensity = self.hawkes_params['mu']  # 重置Hawkes强度
        
        # 滚动波动率计算窗口（15分钟 = 3个5分钟步）
        vol_window = int(15 / (dt * 24 * 60))
        
        for t in range(1, n_steps):
            # 1. 计算对数价格（QED模型作用于对数价格）
            log_x_prev = np.log(prices[t-1])
            # 2. QED扩散更新
            log_x_t = self._qed_diffusion_step(log_x_prev, dt)
            # 3. Hawkes跳跃更新
            jump_occurred, jump_size = self._hawkes_jump_step(dt)
            if jump_occurred:
                log_x_t += jump_size
                has_jump[t] = True
            # 4. 转换为原始价格
            prices[t] = np.exp(log_x_t)
            # 5. 计算收益率
            returns[t] = (prices[t] - prices[t-1]) / prices[t-1]
            # 6. 计算滚动波动率（年化）
            if t >= vol_window:
                window_returns = returns[t-vol_window:t]
                vol_15m[t] = np.std(window_returns) * np.sqrt(24 * 60 / 5 * 365)  # 年化
        
        return pd.DataFrame({
            'time_days': times,
            'price': prices,
            'returns': returns,
            'vol_15m_annual': vol_15m,
            'has_jump': has_jump
        })

# 示例参数（文档推荐的校准值）
DEFAULT_QED_PARAMS = {
    'theta': 0.02 / 365,  # 年化2%净增长率（日频转换）
    'kappa': 1.5,         # 中等均值回归强度
    'g': 0.8,             # 非线性饱和项
    'sigma': 0.4,         # 基础波动率系数（年化40%）
    'omega': 0.2          # 额外冲击系数
}

DEFAULT_HAWKES_PARAMS = {
    'mu': 2.0,            # 日基础跳跃强度（平均2次/天）
    'alpha': 0.3,         # 激发系数（跳跃后强度提升）
    'beta': 0.8,          # 衰减系数（强度恢复速度）
    'jump_mean': -0.02,   # 跳跃均值（负跳为主，2%幅度）
    'jump_std': 0.015     # 跳跃标准差
}

if __name__ == "__main__":
    # 测试模拟器
    simulator = BTCPriceSimulator(
        qed_params=DEFAULT_QED_PARAMS,
        hawkes_params=DEFAULT_HAWKES_PARAMS,
        s0=50000.0,
        seed=42
    )
    # 模拟7天价格路径（5分钟频率）
    path = simulator.simulate_path(T=7.0, dt=1/24/12)
    print("模拟器输出示例：")
    print(path.head(10))
    print(f"\n路径长度：{len(path)} 步")
    print(f"跳跃次数：{path['has_jump'].sum()} 次")
    print(f"最终价格：{path['price'].iloc[-1]:.2f}")
