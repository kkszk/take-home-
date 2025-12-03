import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Dict, Optional
from simulator import BTCPriceSimulator

class OptionPricer:
    """期权定价模块：IV曲面构建 + 布莱克-斯科尔斯定价"""
    
    def __init__(self, 
                 r: float = 0.01,  # 无风险利率（年化）
                 seed: int = 42):
        """
        初始化
        :param r: 无风险利率（年化）
        :param seed: 随机种子
        """
        self.r = r
        self.rng = np.random.default_rng(seed)
    
    def _black_scholes_price(self, 
                            S: float, 
                            K: float, 
                            T: float, 
                            sigma: float, 
                            is_call: bool = True) -> float:
        """
        布莱克-斯科尔斯定价公式
        :param S: 标的资产价格
        :param K: 行权价
        :param T: 剩余到期时间（年）
        :param sigma: 波动率（年化）
        :param is_call: 是否为认购期权（False为认沽）
        :return: 期权理论价格
        """
        if T <= 0:
            return max(S - K, 0.0) if is_call else max(K - S, 0.0)
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price
    
    def _calculate_iv_surface(self, 
                             S: float, 
                             K: float, 
                             T: float, 
                             local_vol: float, 
                             regime: str = 'calm') -> float:
        """
        构建IV曲面（基于文档要求：到期时间、虚值程度、本地波动率、市场状态）
        :param S: 标的价格
        :param K: 行权价
        :param T: 剩余到期时间（年）
        :param local_vol: 本地实现波动率（年化）
        :param regime: 市场状态（calm/volatile）
        :return: 隐含波动率（年化）
        """
        moneyness = K / S  # 虚值程度（K/S=1为平值）
        
        # 1. 基础IV = 本地波动率 + 期限溢价
        term_premium = 0.02 * np.sqrt(T)  # 短期期权期限溢价低
        base_iv = local_vol + term_premium
        
        # 2. 虚值调整（波动率微笑）
        if moneyness < 0.95:  # 深度实值（认沽）
            smile_adjust = 0.03  # 尾部风险溢价
        elif moneyness > 1.05:  # 深度虚值（认购）
            smile_adjust = 0.015
        else:  # 平值附近
            smile_adjust = 0.0
        
        # 3. 市场状态调整（波动 regime 溢价）
        regime_adjust = 0.05 if regime == 'volatile' else 0.0
        
        # 4. 最终IV（限制在合理范围）
        iv = base_iv + smile_adjust + regime_adjust
        return np.clip(iv, 0.2, 1.5)  # IV区间：20% ~ 150%
    
    def estimate_local_vol(self, returns: np.ndarray, window: int = 24) -> float:
        """
        估计本地实现波动率（滚动窗口标准差）
        :param returns: 标的资产收益率序列
        :param window: 滚动窗口大小（默认24步=2小时）
        :return: 年化本地波动率
        """
        if len(returns) < window:
            return 0.4  # 默认值：40%
        recent_returns = returns[-window:]
        vol = np.std(recent_returns)
        return vol * np.sqrt(24 * 60 / 5 * 365)  # 5分钟数据年化
    
    def get_option_price(self, 
                        S: float, 
                        K: float, 
                        T: float, 
                        local_vol: float, 
                        is_call: bool = True, 
                        regime: str = 'calm') -> Tuple[float, float]:
        """
        计算期权价格和IV
        :param S: 标的价格
        :param K: 行权价
        :param T: 剩余到期时间（年）
        :param local_vol: 本地实现波动率
        :param is_call: 是否为认购期权
        :param regime: 市场状态
        :return: (期权价格, 隐含波动率)
        """
        iv = self._calculate_iv_surface(S, K, T, local_vol, regime)
        price = self._black_scholes_price(S, K, T, iv, is_call)
        return price, iv
    
    def generate_option_universe(self, 
                                S: float, 
                                local_vol: float, 
                                regime: str = 'calm') -> pd.DataFrame:
        """
        生成文档规定的期权合约池
        - 行权价：0.9*S0, S0, 1.1*S0（S0为初始价格）
        - 到期日：1d, 3d, 7d, 14d, 1m
        :param S: 当前标的价格
        :param local_vol: 本地波动率
        :param regime: 市场状态
        :return: 期权合约列表（含价格、IV等信息）
        """
        S0 = S  # 假设当前价格为初始价格（可扩展为历史初始价）
        strikes = [0.9 * S0, S0, 1.1 * S0]  # 行权价
        maturities_days = [1, 3, 7, 14, 30]  # 到期日（天）
        option_types = ['call', 'put']
        
        contracts = []
        for strike in strikes:
            for mat_days in maturities_days:
                T = mat_days / 365  # 转换为年
                for opt_type in option_types:
                    is_call = (opt_type == 'call')
                    price, iv = self.get_option_price(
                        S=S,
                        K=strike,
                        T=T,
                        local_vol=local_vol,
                        is_call=is_call,
                        regime=regime
                    )
                    contracts.append({
                        'strike': strike,
                        'maturity_days': mat_days,
                        'time_to_maturity': T,
                        'option_type': opt_type,
                        'moneyness': strike / S,
                        'iv_annual': iv,
                        'price': price
                    })
        return pd.DataFrame(contracts)

if __name__ == "__main__":
    # 测试期权定价模块
    pricer = OptionPricer(r=0.01)
    
    # 1. 单合约定价测试
    S = 50000.0
    K = 50000.0  # 平值
    T = 7 / 365  # 7天到期
    local_vol = 0.4  # 本地波动率40%
    call_price, call_iv = pricer.get_option_price(S, K, T, local_vol, is_call=True, regime='calm')
    put_price, put_iv = pricer.get_option_price(S, K, T, local_vol, is_call=False, regime='calm')
    
    print("单合约定价测试：")
    print(f"平值认购期权价格：{call_price:.2f} USD, IV：{call_iv:.2%}")
    print(f"平值认沽期权价格：{put_price:.2f} USD, IV：{put_iv:.2%}")
    
    # 2. 生成完整期权池
    option_universe = pricer.generate_option_universe(S=50000.0, local_vol=0.4, regime='volatile')
    print(f"\n期权合约池大小：{len(option_universe)} 个合约")
    print("\n合约池示例：")
    print(option_universe[['strike', 'maturity_days', 'option_type', 'moneyness', 'iv_annual', 'price']].head(10))
