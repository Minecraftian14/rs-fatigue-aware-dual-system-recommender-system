# Dynamic Personal Finance Planner

A reinforcement learning framework for dynamic personal financial planning under market and household uncertainty.

This project explores how reinforcement learning can be used to make adaptive portfolio allocation decisions by considering not only financial market conditions but also household-level factors such as income instability, expenses, employment risk, emergency liquidity requirements, and retirement horizons.

The project was completed as an academic research implementation.

## Overview

Traditional portfolio allocation strategies primarily consider investment returns and market risk. However, personal financial decisions are also influenced by changing income, household expenses, employment uncertainty, and financial obligations.

This project models personal finance as a sequential decision-making problem, where an agent dynamically allocates wealth across equity, debt, and cash while accounting for both market conditions and household financial resilience.

The objective is to balance long-term wealth growth with liquidity preservation, downside-risk control, and financial survival.

## Architecture and Components

The implementation consists of the following major components:

| Component | Description and Purpose |
|---|---|
| **Financial Planning Environment** | A Gymnasium-compatible simulation environment that models monthly financial decisions over a finite planning horizon. |
| **Market Dynamics** | Processes historical equity and debt returns, derives market volatility and drawdown indicators, and identifies market regimes (bull, bear, and crisis). Cash returns are modelled separately. |
| **Synthetic User Profiles** | Generates household profiles with varying ages, wealth, income levels, income volatility, employment risk, expenses, emergency-buffer requirements, goal horizons, and risk appetites. |
| **Household Dynamics** | Simulates income growth, promotions, employment shocks, job loss, expense inflation, discretionary spending, and retirement-related income changes. |
| **Portfolio Dynamics** | Updates household wealth based on portfolio returns, income, expenses, and transaction costs. It also tracks portfolio drawdowns and emergency liquidity. |
| **State Representation** | Combines portfolio, market, household, temporal, and user-preference variables into a 17-dimensional observation vector for the RL agent. |
| **Reward Function** | Balances wealth growth against portfolio drawdowns, insufficient emergency liquidity, and financial ruin through a risk-aware reward formulation. |
| **RL Agents** | Implements and evaluates PPO, SAC, TD3, DDPG, and a CVaR-aware SAC variant for continuous portfolio allocation decisions. |
| **Baseline Strategies** | Provides static 60/40 allocation and age-based glide-path strategies for comparison with learned policies. |

## How the Components Connect

The framework follows a monthly interaction loop:

1. **Profile initialization:** A synthetic household profile is sampled, along with an initial wealth level, financial characteristics, and planning horizon.
2. **Market observation:** The environment provides current market returns, volatility, drawdown, and market-regime information.
3. **State construction:** Market conditions are combined with household and portfolio information to construct the agent's observation.
4. **Portfolio allocation:** The RL agent selects allocation weights across equity, debt, and cash.
5. **Financial simulation:** Market returns, income changes, expenses, employment events, and transaction costs are applied to update household wealth.
6. **Reward calculation:** The agent receives a reward reflecting wealth growth, drawdown exposure, emergency-buffer adequacy, and financial survival.
7. **Policy learning:** The process repeats monthly, allowing the agent to learn allocation policies that account for changing financial circumstances.

The action space is continuous, with a softmax transformation ensuring that portfolio weights are non-negative and sum to one.

## Reinforcement Learning Approaches

The project implements and compares five reinforcement learning approaches:

- **PPO (Proximal Policy Optimization):** An on-policy actor-critic algorithm for continuous allocation decisions.
- **SAC (Soft Actor-Critic):** An off-policy algorithm incorporating entropy-based exploration.
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient):** A deterministic actor-critic approach designed to improve learning stability.
- **DDPG (Deep Deterministic Policy Gradient):** An off-policy deterministic policy-gradient algorithm.
- **CVaR-aware SAC:** A SAC variant incorporating lower-tail reward adjustments to increase sensitivity to adverse outcomes.

These approaches are evaluated against two conventional allocation strategies:

- **Static 60/40:** Allocates the non-emergency-reserve portion of wealth between equity and debt using a fixed 60:40 ratio.
- **Age-based Glide Path:** Gradually reduces equity exposure as the investor ages.

## Evaluation

The learned policies and baseline strategies are compared using both portfolio-performance and financial-resilience metrics:

- Average and median final wealth
- Wealth standard deviation
- Sharpe ratio
- Maximum drawdown
- Ruin probability
- Conditional Value at Risk (CVaR) of final wealth at the 10% lower tail

The evaluation considers multiple simulated household trajectories and market conditions, highlighting the trade-offs between wealth accumulation, consistency, liquidity, and downside risk.

## Technologies and Methods

- **Python**
- **Reinforcement Learning:** Stable-Baselines3
- **Simulation Environment:** Gymnasium
- **Algorithms:** PPO, SAC, TD3, DDPG
- **Numerical Computing and Data Processing:** NumPy, Pandas
- **Core Concepts:** Portfolio optimization, continuous control, Markov Decision Processes, household financial simulation, risk-sensitive reinforcement learning.

## Scope and Limitations

This repository contains an experimental research implementation of a simulated personal financial planning framework. It is intended to explore and compare adaptive portfolio allocation strategies under different market and household conditions.

Household profiles and financial events are synthetically generated, and the framework relies on simplified behavioural and economic assumptions. The results therefore represent simulation-based experiments rather than validated financial advice or recommendations for real-world investment decisions.

The project is not a production-ready financial planning application.
