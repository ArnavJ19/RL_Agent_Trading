
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnv, self).__init__()

        self.data = data
        self.current_step = 0
        
        # Action space: Continuous actions - range between -1 and 1
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: Use all trends (indicators) from the dataset
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(self.data.columns),), dtype=np.float32)
        
        # Initial balance and shares
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.portfolio_value_history = []
        self.returns = []
        self.trade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.portfolio_value_history = [self.net_worth]
        self.returns = []
        self.trade_log = []
        return self._next_observation(), {}

    def _next_observation(self):
        observation = self.data.iloc[self.current_step].values
        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation)
        return observation
    
    def _take_action(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        current_date = self.data.index[self.current_step]

        action_value = action[0]
        if action_value > 0:
            buy_amount = self.balance * action_value
            shares_to_buy = buy_amount // current_price
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.trade_log.append({
                    'step': self.current_step,
                    'date': current_date,
                    'action': 'Buy',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'balance': self.balance
                })

        elif action_value < 0:
            sell_amount = self.shares_held * -action_value
            if sell_amount > 0:
                shares_to_sell = min(sell_amount, self.shares_held)
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.trade_log.append({
                    'step': self.current_step,
                    'date': current_date,
                    'action': 'Sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'balance': self.balance
                })

        self.net_worth = self.shares_held * current_price + self.balance
        self.portfolio_value_history.append(self.net_worth)
        
        if len(self.portfolio_value_history) > 1:
            ret = (self.net_worth - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]
            self.returns.append(ret)

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward()
        obs = self._next_observation()
        return obs, reward, done, {}, {}

    def _calculate_reward(self):
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close'] if self.current_step + 1 < len(self.data) else current_price
        net_worth_before = self.net_worth
        net_worth_after = self.shares_held * next_price + self.balance
        reward = net_worth_after - net_worth_before

        if len(self.returns) > 1:
            avg_return = np.mean(self.returns)
            return_std = np.std(self.returns)
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            reward += sharpe_ratio
        
        reward = np.tanh(reward / 1000)
        return reward
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Shares Held: {self.shares_held}, Balance: {self.balance}')
    
    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)
