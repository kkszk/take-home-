import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
from simulator import BTCPriceSimulator, DEFAULT_QED_PARAMS, DEFAULT_HAWKES_PARAMS
from option_surface import OptionPricer
from mm_strategy import BTCMarketMaker

class RLOptionHedgingEnv(gym.Env):
    """强化学习对冲环境（MDP框架）：做市库存 + 期权对冲"""
    
    metadata = {"render_modes": ["human", "logs"], "render_fps": 1}
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 2016,  # 7天 * 24小时 * 12步/小时（5分钟）
                 mm_params: dict = None,
                 simulator_params: dict = None,
                 pricer_params: dict = None):
        """
        初始化环境
        :param render_mode: 渲染模式（human/logs）
        :param max_episode_steps: 每轮最大步数
        :param mm_params: 做市策略参数
        :param simulator_params: 价格模拟器参数
        :param pricer_params: 期权定价参数
        """
        super().__init__()
        
        # 1. 初始化子模块参数
        self.mm_params = mm_params or {}
        self.simulator_params = simulator_params or {}
        self.pricer_params = pricer_params or {}
        
        # 2. 初始化核心模块
        self.simulator = self._init_simulator()
        self.pricer = self._init_pricer()
        self.market_maker = self._init_mm()
        
        # 3. 环境配置
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.render_mode = render_mode
        
        # 4. 期权合约池（文档规定的合约）
        self.option_universe = None
        self.strike_list = [0.9, 1.0, 1.1]  # 相对于初始价格的比例
        self.maturity_list = [1, 3, 7, 14, 30]  # 到期日（天）
        
        # 5. 动作空间：对冲动作（每个期权合约的买卖数量，含无动作）
        # 动作定义：[call_0.9_1d, put_0.9_1d, ..., call_1.1_30d, put_1.1_30d, do_nothing]
        self.n_options = len(self.strike_list) * len(self.maturity_list) * 2  # 3*5*2=30个合约
        self.action_space = spaces.Discrete(self.n_options + 1)  # +1为无动作
        
        # 6. 观测空间：文档规定的状态向量
        self.observation_space = spaces.Box(
            low=np.array([
                10000.0,   # S_t: BTC价格下限
                -self.market_maker.I_max,  # I_t: 做市库存下限
                -1.0,      # H_t: 对冲仓位下限（标准化）
                0.0,       # V_t: 期权组合价值下限
                0.1,       # 本地波动率下限
                0.0,       # TTM: 剩余到期时间下限
                -0.5,      # moneyness: 虚值程度下限（log(K/S)）
                -0.1,      # ΔS_t: 短期价格变化下限
                -0.05      # ΔV_t: 期权价值变化下限
            ]),
            high=np.array([
                100000.0,  # S_t: BTC价格上限
                self.market_maker.I_max,   # I_t: 做市库存上限
                1.0,       # H_t: 对冲仓位上限（标准化）
                10000.0,   # V_t: 期权组合价值上限
                1.5,       # 本地波动率上限
                30/365,    # TTM: 剩余到期时间上限
                0.5,       # moneyness: 虚值程度上限
                0.1,       # ΔS_t: 短期价格变化上限
                0.05       # ΔV_t: 期权价值变化上限
            ]),
            dtype=np.float32
        )
        
        # 7. 环境状态变量
        self.state = {
            'S_t': 0.0,          # BTC当前价格
            'I_t': 0.0,          # 做市库存（BTC）
            'H_t': 0.0,          # 期权对冲仓位（标准化）
            'V_t': 0.0,          # 期权组合价值（USD）
            'local_vol': 0.0,    # 本地实现波动率（年化）
            'TTM': 0.0,          # 期权剩余到期时间（年）
            'moneyness': 0.0,    # 虚值程度（log(K/S)）
            'delta_S': 0.0,      # 短期价格变化（S_t - S_{t-1}）/ S_{t-1}
            'delta_V': 0.0       # 期权价值变化（V_t - V_{t-1}）/ V_{t-1}
        }
        self.prev_state = None  # 上一步状态（用于计算ΔS和ΔV）
        self.hedge_positions = {}  # 当前期权对冲仓位（{合约ID: 数量}）
        self.option_transaction_cost = 0.0005  # 期权交易成本（0.05% of notional）
    
    def _init_simulator(self) -> BTCPriceSimulator:
        """初始化价格模拟器"""
        qed_params = self.simulator_params.get('qed_params', DEFAULT_QED_PARAMS)
        hawkes_params = self.simulator_params.get('hawkes_params', DEFAULT_HAWKES_PARAMS)
        s0 = self.simulator_params.get('s0', 50000.0)
        seed = self.simulator_params.get('seed', 42)
        return BTCPriceSimulator(qed_params, hawkes_params, s0, seed)
    
    def _init_pricer(self) -> OptionPricer:
        """初始化期权定价器"""
        r = self.pricer_params.get('r', 0.01)
        seed = self.pricer_params.get('seed', 42)
        return OptionPricer(r, seed)
    
    def _init_mm(self) -> BTCMarketMaker:
        """初始化做市商"""
        s0 = self.mm_params.get('s0', 0.001)
        q0 = self.mm_params.get('q0', 0.1)
        I_max = self.mm_params.get('I_max', 1.0)
        Pi_min = self.mm_params.get('Pi_min', 50000.0)
        seed = self.mm_params.get('seed', 42)
        return BTCMarketMaker(s0, q0, I_max, Pi_min, seed)
    
    def _get_market_regime(self, local_vol: float) -> str:
        """根据本地波动率判断市场状态（calm/volatile）"""
        return 'volatile' if local_vol > 0.6 else 'calm'  # 60%为阈值
    
    def _update_hedge_value(self) -> float:
        """计算当前期权对冲组合的总价值"""
        total_value = 0.0
        S_t = self.state['S_t']
        local_vol = self.state['local_vol']
        regime = self._get_market_regime(local_vol)
        
        for contract_id, quantity in self.hedge_positions.items():
            if quantity == 0:
                continue
            # 解析合约ID（格式：call_0.9_1d）
            opt_type, strike_ratio, mat_days = contract_id.split('_')
            strike = float(strike_ratio) * self.simulator.s0
            T = int(mat_days.replace('d', '')) / 365
            
            # 计算当前期权价格
            price, _ = self.pricer.get_option_price(
                S=S_t,
                K=strike,
                T=T,
                local_vol=local_vol,
                is_call=(opt_type == 'call'),
                regime=regime
            )
            total_value += quantity * price
        return total_value
    
    def _get_observation(self) -> np.ndarray:
        """构建观测向量（文档规定的状态变量）"""
        return np.array([
            self.state['S_t'],
            self.state['I_t'],
            self.state['H_t'],
            self.state['V_t'],
            self.state['local_vol'],
            self.state['TTM'],
            self.state['moneyness'],
            self.state['delta_S'],
            self.state['delta_V']
        ], dtype=np.float32)
    
    def _calculate_reward(self, prev_equity: float) -> float:
        """
        计算奖励函数（文档推荐）
        奖励 = 权益变化 - 交易成本 - 风险惩罚
        """
        # 1. 总权益变化（做市权益 + 期权对冲价值）
        current_equity = self.market_maker.Pi_t + self.state['V_t']
        delta_equity = current_equity - prev_equity
        
        # 2. 期权交易成本（0.05% of notional）
        transaction_cost = self.state.get('transaction_cost', 0.0)
        
        # 3. 风险惩罚（库存+对冲仓位的平方惩罚，避免过度暴露）
        net_position = self.state['I_t'] + self.state['H_t'] * self.market_maker.I_max
        risk_penalty = 0.1 * (net_position ** 2)
        
        # 最终奖励
        reward = delta_equity - transaction_cost - risk_penalty
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        环境单步交互
        :param action: 代理动作（期权对冲动作）
        :return: (观测, 奖励, 终止, 截断, 信息)
        """
        self.current_step += 1
        prev_equity = self.market_maker.Pi_t + self.state['V_t']
        self.prev_state = self.state.copy()
        
        # 1. 生成新的价格（从模拟器获取）
        if self.current_step == 1:
            # 第一步：生成完整路径并缓存
            self.price_path = self.simulator.simulate_path(
                T=self.max_episode_steps * (5/60/24),  # 总时长=步数*5分钟
                dt=5/60/24  # 5分钟步长
            )
        S_new = self.price_path.iloc[self.current_step]['price']
        returns = self.price_path['returns'].iloc[:self.current_step+1].values
        
        # 2. 做市商单步模拟（更新库存和现金）
        mm_quotes, mm_trade = self.market_maker.simulate_step(S_new)
        
        # 3. 执行期权对冲动作
        transaction_cost = 0.0
        if action < self.n_options:  # 执行期权交易（非无动作）
            # 解析动作对应的合约
            opt_idx = action // 2
            is_call = (action % 2 == 0)
            strike_idx = opt_idx // (len(self.maturity_list))
            mat_idx = opt_idx % len(self.maturity_list)
            
            strike_ratio = self.strike_list[strike_idx]
            mat_days = self.maturity_list[mat_idx]
            strike = strike_ratio * self.simulator.s0
            contract_id = f"{'call' if is_call else 'put'}_{strike_ratio}_{mat_days}d"
            
            # 对冲仓位大小（固定为0.01 BTC，可调整）
            hedge_size = 0.01
            # 计算名义价值（交易成本基数）
            notional = hedge_size * S_new
            # 扣除交易成本
            transaction_cost = notional * self.option_transaction_cost
            self.market_maker.Cash_t -= transaction_cost  # 从现金中扣除
            
            # 更新对冲仓位
            if contract_id not in self.hedge_positions:
                self.hedge_positions[contract_id] = 0.0
            self.hedge_positions[contract_id] += hedge_size if is_call else -hedge_size
        
        # 4. 更新状态变量
        local_vol = self.pricer.estimate_local_vol(returns)
        regime = self._get_market_regime(local_vol)
        self.option_universe = self.pricer.generate_option_universe(S_new, local_vol, regime)
        
        # 选择平值7天期权作为参考（状态变量中的TTM和moneyness）
        ref_option = self.option_universe[
            (self.option_universe['moneyness'] >= 0.98) & 
            (self.option_universe['moneyness'] <= 1.02) & 
            (self.option_universe['maturity_days'] == 7)
        ].iloc[0]
        
        # 计算状态变量
        self.state['S_t'] = S_new
        self.state['I_t'] = self.market_maker.I_t
        self.state['H_t'] = sum(self.hedge_positions.values()) / self.market_maker.I_max  # 标准化
        self.state['V_t'] = self._update_hedge_value()
        self.state['local_vol'] = local_vol
        self.state['TTM'] = ref_option['time_to_maturity']
        self.state['moneyness'] = np.log(ref_option['moneyness'])
        self.state['delta_S'] = (S_new - self.prev_state['S_t']) / self.prev_state['S_t'] if self.current_step > 1 else 0.0
        self.state['delta_V'] = (self.state['V_t'] - self.prev_state['V_t']) / self.prev_state['V_t'] if self.current_step > 1 and self.prev_state['V_t'] != 0 else 0.0
        self.state['transaction_cost'] = transaction_cost
        
        # 5. 计算奖励
        reward = self._calculate_reward(prev_equity)
        
        # 6. 检查终止条件
        terminated = (self.market_maker.Pi_t < self.market_maker.Pi_min)  # 权益过低
        truncated = (self.current_step >= self.max_episode_steps)  # 步数用尽
        
        # 7. 渲染（可选）
        if self.render_mode == 'human':
            self._render_human()
        
        # 8. 信息字典
        info = {
            'step': self.current_step,
            'price': S_new,
            'mm_inventory': self.market_maker.I_t,
            'hedge_positions': self.hedge_positions.copy(),
            'total_equity': self.market_maker.Pi_t + self.state['V_t'],
            'transaction_cost': transaction_cost,
            'market_regime': regime
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境到初始状态"""
        super().reset(seed=seed, options=options)
        
        # 1. 重置子模块
        self.market_maker.reset(initial_price=self.simulator.s0, initial_cash=100000.0)
        self.current_step = 0
        self.hedge_positions = {}
        
        # 2. 初始化状态
        S0 = self.simulator.s0
        self.state = {
            'S_t': S0,
            'I_t': 0.0,
            'H_t': 0.0,
            'V_t': 0.0,
            'local_vol': 0.4,  # 初始本地波动率40%
            'TTM': 7/365,       # 初始参考期权到期时间（7天）
            'moneyness': 0.0,   # 平值期权
            'delta_S': 0.0,
            'delta_V': 0.0,
            'transaction_cost': 0.0
        }
        self.prev_state = self.state.copy()
        
        # 3. 生成初始期权池
        self.option_universe = self.pricer.generate_option_universe(
            S=S0,
            local_vol=0.4,
            regime='calm'
        )
        
        # 4. 渲染（可选）
        if self.render_mode == 'human':
            print("环境重置完成，初始状态：")
            print(f"初始价格：{S0:.2f} USD, 初始权益：{self.market_maker.Pi_t:.2f} USD")
        
        return self._get_observation(), {'initial_price': S0, 'initial_equity': self.market_maker.Pi_t}
    
    def _render_human(self):
        """人类可读渲染"""
        if self.current_step % 100 == 0:  # 每100步输出一次
            print(f"\n=== 第{self.current_step}步 ===")
            print(f"价格：{self.state['S_t']:.2f} USD")
            print(f"做市库存：{self.state['I_t']:.4f} BTC")
            print(f"对冲仓位（标准化）：{self.state['H_t']:.4f}")
            print(f"期权组合价值：{self.state['V_t']:.2f} USD")
            print(f"总权益：{self.market_maker.Pi_t + self.state['V_t']:.2f} USD")
            print(f"本地波动率：{self.state['local_vol']:.2%}")

if __name__ == "__main__":
    # 测试RL环境
    env = RLOptionHedgingEnv(render_mode='human')
    obs, info = env.reset()
    
    print("\n初始观测：")
    print(obs)
    print(f"初始信息：{info}")
    
    # 模拟5步交互
    for _ in range(5):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n动作：{action}")
        print(f"观测：{obs}")
        print(f"奖励：{reward:.2f}")
        print(f"信息：{info}")
        if terminated or truncated:
            break
    
    env.close()
