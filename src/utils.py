import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from typing import Tuple, List, Dict, Optional
import json
import os

def plot_stylized_facts(returns: np.ndarray, price_path: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制BTC收益率的典型事实（文档要求）
    1. 收益率直方图 + QQ图
    2. 滚动实现波动率（15m, 1h, 4h）
    3. 收益率和绝对收益率的自相关函数
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("BTC Perpetual Stylized Facts", fontsize=16)
    
    # 1. 收益率直方图
    axes[0,0].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title("Returns Histogram (Fat Tails)")
    axes[0,0].set_xlabel("Returns")
    axes[0,0].set_ylabel("Frequency")
    axes[0,0].axvline(returns.mean(), color='red', linestyle='--', label=f"Mean: {returns.mean():.4f}")
    axes[0,0].legend()
    
    # 2. QQ图（对比正态分布）
    probplot(returns, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("QQ-Plot (Deviation from Normality)")
    
    # 3. 滚动实现波动率
    # 转换为5分钟数据的窗口大小：15m=3步，1h=12步，4h=48步
    windows = [3, 12, 48]
    labels = ['15m', '1h', '4h']
    colors = ['green', 'orange', 'red']
    
    for window, label, color in zip(windows, labels, colors):
        vol = price_path['returns'].rolling(window=window).std() * np.sqrt(24 * 60 / 5 * 365)  # 年化
        axes[1,0].plot(price_path['time_days'], vol, label=f"Rolling Vol ({label})", color=color, linewidth=1.5)
    
    axes[1,0].set_title("Rolling Realized Volatility (Clustering)")
    axes[1,0].set_xlabel("Time (Days)")
    axes[1,0].set_ylabel("Annualized Volatility")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 自相关函数（ACF）
    max_lag = 50
    acf_returns = [autocorrelation(returns, lag) for lag in range(max_lag)]
    acf_abs_returns = [autocorrelation(np.abs(returns), lag) for lag in range(max_lag)]
    
    axes[1,1].bar(range(max_lag), acf_returns, alpha=0.7, label='Returns ACF', color='blue', width=0.4)
    axes[1,1].bar(np.array(range(max_lag)) + 0.4, acf_abs_returns, alpha=0.7, label='Abs Returns ACF', color='red', width=0.4)
    axes[1,1].set_title("Autocorrelation Function (Volatility Clustering)")
    axes[1,1].set_xlabel("Lag")
    axes[1,1].set_ylabel("ACF Value")
    axes[1,1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def autocorrelation(x: np.ndarray, lag: int) -> float:
    """计算序列的自相关系数"""
    if lag == 0:
        return 1.0
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def calculate_metrics(equity_paths: List[np.ndarray]) -> Dict[str, float]:
    """
    计算策略评估指标（文档要求）
    - 夏普比率（年化）
    - 最大回撤
    - 条件风险价值（CVaR，95%置信度）
    - 平均收益率
    - 收益率标准差
    """
    # 计算每日收益率（假设每条路径是按天采样）
    daily_returns = []
    for path in equity_paths:
        # 假设路径是按5分钟采样，转换为日收益率（每天24*12=288步）
        daily_steps = 288
        for i in range(daily_steps, len(path), daily_steps):
            daily_ret = (path[i] - path[i-daily_steps]) / path[i-daily_steps]
            daily_returns.append(daily_ret)
    daily_returns = np.array(daily_returns)
    
    # 夏普比率（无风险利率1%）
    risk_free_rate = 0.01
    sharpe = (np.mean(daily_returns) * 365 - risk_free_rate) / (np.std(daily_returns) * np.sqrt(365))
    
    # 最大回撤
    max_drawdowns = []
    for path in equity_paths:
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        max_drawdowns.append(np.min(drawdown))
    max_drawdown = np.mean(max_drawdowns)
    
    # CVaR（95%置信度）
    cvar = np.percentile(daily_returns, 5)
    
    # 其他指标
    avg_return = np.mean(daily_returns) * 365  # 年化平均收益率
    return_std = np.std(daily_returns) * np.sqrt(365)  # 年化收益率标准差
    
    return {
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'cvar_95': float(cvar),
        'annualized_return': float(avg_return),
        'annualized_volatility': float(return_std),
        'num_paths': len(equity_paths)
    }

def save_metrics(metrics: Dict[str, float], save_path: str):
    """保存评估指标到JSON文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")

def load_metrics(load_path: str) -> Dict[str, float]:
    """从JSON文件加载评估指标"""
    with open(load_path, 'r') as f:
        return json.load(f)

def plot_equity_paths(equity_paths: List[np.ndarray], 
                     labels: List[str], 
                     save_path: Optional[str] = None):
    """绘制多条权益路径对比（RL对冲 vs 基准策略）"""
    plt.figure(figsize=(12, 6))
    for path, label in zip(equity_paths, labels):
        plt.plot(path, label=label, alpha=0.7, linewidth=1.5)
    plt.title("Equity Paths Comparison (RL Hedging vs Baseline)")
    plt.xlabel("Steps")
    plt.ylabel("Total Equity (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_inventory_behavior(inventory_paths: List[np.ndarray], 
                           labels: List[str], 
                           save_path: Optional[str] = None):
    """绘制库存行为对比"""
    plt.figure(figsize=(12, 6))
    for path, label in zip(inventory_paths, labels):
        plt.plot(path, label=label, alpha=0.7, linewidth=1.5)
    plt.title("Inventory Behavior (RL Hedging vs Baseline)")
    plt.xlabel("Steps")
    plt.ylabel("Inventory (BTC)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 测试工具函数
    from simulator import BTCPriceSimulator, DEFAULT_QED_PARAMS, DEFAULT_HAWKES_PARAMS
    
    # 设置随机种子
    set_random_seeds(42)
    
    # 生成测试数据
    simulator = BTCPriceSimulator(DEFAULT_QED_PARAMS, DEFAULT_HAWKES_PARAMS, s0=50000.0)
    price_path = simulator.simulate_path(T=7.0)
    returns = price_path['returns'].values[1:]  # 去除第一个0
    
    # 绘制典型事实
    plot_stylized_facts(returns, price_path, save_path="../results/plots/stylized_facts.png")
    
    # 计算指标
    equity_paths = [price_path['price'].values * 0.1 + 100000]  # 模拟权益路径
    metrics = calculate_metrics(equity_paths)
    save_metrics(metrics, "../results/metrics.json")
    print("测试指标：", metrics)
