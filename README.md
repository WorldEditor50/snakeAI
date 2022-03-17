# snakeAI
- A* agent: a method bases on greedy policy
- random search agent: a method bases on reward and MCMC
- bp agent: learn from A* agent
- dqn agent: DQN
- dpg agent: policy gradient 
- ddpg agent:  todo
- PPO agent: learn with adaptive KL-penalty and clipped objective
- lstm + q-learning agent: generate sequence by sampling data from replay buffer with uniform distribution and backpropagate error through time with EMA
- DRPG agent: lstm + policy gradient

## 1. Markov Decision Process

### definition:
- **{St, At, P(St+1|St, At), R, St+1, γ}**
- **St** --- state
- **At** --- action
- **P(St+1|St, At)** --- transition probability
- **R** --- reward
- **St+1** --- next state
- **γ** --- discounted factor

### Process:
  agent is in the state **St** , taking action **At** , then  transitioning into state **St+1** and getting reward **R** in the state **St+1**
### Assumption of the Markov Property:
  the effects of an action taken in a state depend only on that state and not on the prior history.



## 2. Value Function

### 2.1 state value function

**Gt = Rt+1 + γ * Rt+2 + γ^2 * Rt+3 + ... = ∑γ^k * Rt+k+1**

**Gt = Rt+1 + γ * Gt+1**

**Vπ(s) = Eπ[Rt+1 + γ * Vπ(St+1) | St = s]**

### 2.2 action value function

**Qπ(s, a) = Eπ[Rt+1 + γ * Qπ(St+1, At+1) | St = s, At = a]**

## 3. Optimization: maximizing cumulative future reward

**θ = argmax E[∑R(St, At)]** 

## 4. sampling method

### 4.1 MCMC

### 4.2 TD
