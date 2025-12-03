import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from simulator import BTCPriceSimulator
from option_surface import OptionPricer

class BTCMarketMaker:
    """BTC永续合约做市策略（文档规定的基准策略）"""
    
    def __init__(self, 
                 s0: float = 0.001,  # 基础点差（0.1%）
                 q0: float = 0.1,    # 基础挂单量（BTC）
                 I_max: float = 1.0, # 最大库存限制（BTC）
                 Pi_min: float = 10000.0,  # 最低权益限制（USD）
                 seed: int = 42):
        """
        初始化做市策略参数
        :param s0: 基础半挂单价差（如0.001=0.1%）
        :param q0: 基础挂单量（BTC）
        :param I_max: 最大允许库存（BTC）
        :param Pi_min: 最低权益（USD）
        :param seed: 随机种子
        """
        self.s0 = s0
        self.q0 = q0
        self.I_max = I_max
        self.Pi_min = Pi_min
        self.rng = np.random.default_rng(seed)
        
        # 策略状态变量
        self.S_t = 0.0  # 当前标的价格
        self.I_t = 0.0  # 当前库存（BTC）
        self.Cash_t = 0.0  # 当前现金（USD）
        self.Pi_t = 0.0  # 当前权益（USD）
        self.phi_t = 0.0  # 标准化库存（I_t / I_max）
    
    def _update_state(self, S_new: float):
        """更新策略状态（价格、权益、标准化库存）"""
        self.S_t = S_new
        self.Pi_t = self.Cash_t + self.I_t * self.S_t
        self.phi_t = self.I_t / self.I_max if self.I_max != 0 else 0.0
    
    def get_quotes(self) -> Dict[str, float]:
        """
        生成挂单报价（根据库存调整价差和挂单量）
        :return: 买单价格、卖单价格、买单量、卖单量
        """
        # 1. 基础报价（无库存偏差）
        bid_price_base = self.S_t * (1 - self.s0)
        ask_price_base = self.S_t * (1 + self.s0)
        
        # 2. 库存调整价差（库存偏多时降低卖价、提高买价，加速去库存）
        k_s = 0.5 * self.s0  # 库存价差调整系数（文档推荐）
        bid_price = bid_price_base - self.S_t * k_s * self.phi_t
        ask_price = ask_price_base - self.S_t * k_s * self.phi_t
        
        # 3. 库存调整挂单量（库存越满，挂单量越小）
        order_size = self.q0 * max(0.0, 1 - abs(self.phi_t))
        
        # 4. 强制去库存开关（库存超限时只挂反向单）
        if abs(self.I_t) >= self.I_max:
            if self.I_t > 0:  # 库存过多，只挂卖单
                return {'bid_price': 0.0, 'ask_price': ask_price, 'bid_size': 0.0, 'ask_size': order_size}
            else:  # 库存过少（空头过多），只挂买单
                return {'bid_price': bid_price, 'ask_price': 0.0, 'bid_size': order_size, 'ask_size': 0.0}
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': order_size,
            'ask_size': order_size
        }
    
    def execute_trade(self, trade_type: str, trade_size: float):
        """
        执行交易（更新库存和现金）
        :param trade_type: 交易类型（buy/sell）
        :param trade_size: 交易数量（BTC）
        """
        if trade_type == 'buy':
            # 买入：现金减少，库存增加
            cost = trade_size * self.S_t
            self.Cash_t -= cost
            self.I_t += trade_size
        elif trade_type == 'sell':
            # 卖出：现金增加，库存减少
            revenue = trade_size * self.S_t
            self.Cash_t += revenue
            self.I_t -= trade_size
        else:
            raise ValueError("交易类型必须为 'buy' 或 'sell'")
        
        # 交易后更新状态
        self._update_state(self.S_t)
    
    def simulate_step(self, S_new: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        单步做市模拟（更新价格、生成报价、随机成交）
        :param S_new: 新的标的价格
        :return: 报价信息、交易结果
        """
        # 1. 更新状态（新价格）
        self._update_state(S_new)
        
        # 2. 检查风险限制（权益过低时停止做市）
        if self.Pi_t < self.Pi_min:
            return {'bid_price': 0.0, 'ask_price': 0.0, 'bid_size': 0.0, 'ask_size': 0.0}, {'trade_type': 'none', 'trade_size': 0.0}
        
        # 3. 生成报价
        quotes = self.get_quotes()
        
        # 4. 随机成交（简化：根据报价与市场价格的偏离度决定成交概率）
        trade_result = {'trade_type': 'none', 'trade_size': 0.0}
        
        # 买单成交条件：市场价格 ≤ 挂单买价
        if quotes['bid_price'] > 0 and S_new <= quotes['bid_price']:
            trade_prob = 0.3  # 成交概率（可根据流动性调整）
            if self.rng.random() < trade_prob:
                trade_size = quotes['bid_size'] * self.rng.uniform(0.5, 1.0)  # 部分成交
                self.execute_trade('buy', trade_size)
                trade_result = {'trade_type': 'buy', 'trade_size': trade_size}
        
        # 卖单成交条件：市场价格 ≥ 挂单卖价
        elif quotes['ask_price'] > 0 and S_new >= quotes['ask_price']:
            trade_prob = 0.3
            if self.rng.random() < trade_prob:
                trade_size = quotes['ask_size'] * self.rng.uniform(0.5, 1.0)
                self.execute_trade('sell', trade_size)
                trade_result = {'trade_type': 'sell', 'trade_size': trade_size}
        
        return quotes, trade_result
    
    def reset(self, initial_price: float, initial_cash: float = 100000.0):
        """重置策略状态（用于蒙特卡洛模拟）"""
        self.S_t = initial_price
        self.I_t = 0.0
        self.Cash_t = initial_cash
        self.Pi_t = initial_cash + self.I_t * self.S_t
        self.phi_t = 0.0

if __name__ == "__main__":
    # 测试做市策略
    mm = BTCMarketMaker(
        s0=0.001,  # 0.1% 半价差
        q0=0.1,    # 0.1 BTC 基础挂单量
        I_max=1.0, # 最大1 BTC 库存
        Pi_min=50000.0  # 最低5万 USD 权益
    )
    
    # 模拟价格序列（5个价格点）
    price_series = [50000.0, 50050.0, 49980.0, 50100.0, 50020.0]
    
    print("做市策略模拟结果：")
    print(f"初始状态 - 权益：{mm.Pi_t:.2f} USD, 库存：{mm.I_t:.4f} BTC, 现金：{mm.Cash_t:.2f} USD")
    
    for i, price in enumerate(price_series[1:], 1):
        quotes, trade = mm.simulate_step(price)
        print(f"\n第{i}步 - 价格：{price:.2f} USD")
        print(f"  报价：买单 {quotes['bid_price']:.2f} USD ({quotes['bid_size']:.4f} BTC), 卖单 {quotes['ask_price']:.2f} USD ({quotes['ask_size']:.4f} BTC)")
        print(f"  交易：{trade['trade_type']} {trade['trade_size']:.4f} BTC")
        print(f"  状态：权益 {mm.Pi_t:.2f} USD, 库存 {mm.I_t:.4f} BTC, 现金 {mm.Cash_t:.2f} USD")
