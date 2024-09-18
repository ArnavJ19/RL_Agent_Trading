# RL_Agent_Trading
This repository contains the implementation of a Reinforcement Learning (RL) trading agent that utilizes Proximal Policy Optimization (PPO), built using Stable Baselines3. The agent is designed to dynamically adjust its portfolio based on market signals and trends from S&amp;P 500 data. 

# Key Features:
1. RL Algorithm: Proximal Policy Optimization (PPO) with an MLP-based policy.
2. Environment: Custom trading environment using S&P 500 data.
3. Comparison: The RL agent's performance is compared to a Buy-and-Hold strategy across several metrics:
4. Cumulative Returns: The RL agent delivered over 15% higher cumulative returns.
5. Sharpe Ratio: The RL agent demonstrated approximately 40% higher risk-adjusted returns.
6. Portfolio Value: The RL agent’s portfolio value was 10% higher by the end of the evaluation period.
7. Volatility Management: While taking on slightly more risk, the RL agent managed volatility better during downturns and capitalized on market upswings.

# Project Highlights:
1. Active Portfolio Management: The RL agent adapts dynamically to market conditions, executing frequent buy/sell decisions based on the policy learned from historical data.
2. Superior Risk-Adjusted Performance: The agent achieved a higher Sharpe ratio compared to the Buy-and-Hold strategy, showing better risk management and return optimization.
3. Data-Driven Strategy: By leveraging historical market data, the agent was able to make informed trading decisions to outperform the static Buy-and-Hold strategy.
