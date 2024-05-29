# snakeAI

## 1. Features

- A* agent

  a method based on greedy policy

- random search agent

   a method based on reward and MCMC

- DQN

- DPG

- PPO

- SAC

- Tanh-Norm

  an approximation of RMS-Norm, with a *better* *robust* performance in off-policy learning

  

## 2. Tricks 

- initialize weight in uniform distribution U(-1, 1)

- use RMSProp as optimizer

- use layer-norm in on-policy methods

- use weight-decay in on-policy methods

- normalize gradient

- the range of reward is symmetric and  the value falls between -1 and 1 

  

## 3. Reward

- reward at position

```c++
float Agent::reward0(int xi, int yi, int xn, int yn, int xt, int yt)
{
    /* agent goes out of the map */
    if (map(xn, yn) == 1) {
        return -1;
    }
    /* agent reaches to the target's position */
    if (xn == xt && yn == yt) {
        return 1;
    }
    /* the distance from agent's previous position to the target's position */
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    /* the distance from agent's current position to the target's position */
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return std::sqrt(d1) - std::sqrt(d2);
}
```

-  cumulative reward per epoch

    DQN reward

  ![dqn-reward](https://github.com/WorldEditor50/snakeAI/raw/master/reward.png) 
  
  the reward agent received will be decreased when agent gets closer to the target until it reaches to the target's position. Otherwise, the agent will tend to be overconfident.