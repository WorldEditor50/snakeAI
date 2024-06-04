#ifndef AGENT_H
#define AGENT_H
#include <vector>
#include "rl/dqn.h"
#include "rl/dpg.h"
#include "rl/ddpg.h"
#include "rl/ppo.h"
#include "rl/qlstm.h"
#include "rl/drpg.h"
#include "rl/convpg.h"
#include "rl/convdqn.h"
#include "rl/tensor.hpp"
#include "rl/sac.h"
#include "common.h"
#include "snake.h"

class Environment;

class Agent
{
private:
    Environment &env;
    Snake &snake;
    RL::Tensor state;
    RL::Tensor nextState;
    RL::DQN dqn;
    RL::BPNN bpnn;
    RL::DPG dpg;
    RL::DDPG ddpg;
    RL::PPO ppo;
    RL::SAC sac;
    RL::QLSTM qlstm;
    RL::DRPG drpg;
    RL::ConvPG convpg;
    RL::ConvDQN convdqn;
    bool trainFlag;
public:
    explicit Agent(Environment& env, Snake& s);
    ~Agent();
    void observe(RL::Tensor& statex, int x, int y, int xt, int yt);
    int astarAction(int x, int y, int xt, int yt, float &totalReward);
    int randAction(int x, int y, int xt, int yt, float &totalReward);
    int dqnAction(int x, int y, int xt, int yt, float &totalReward);
    int qlstmAction(int x, int y, int xt, int yt, float &totalReward);
    int dpgAction(int x, int y, int xt, int yt, float &totalReward);
    int drpgAction(int x, int y, int xt, int yt, float &totalReward);
    int convpgAction(int x, int y, int xt, int yt, float &totalReward);
    int convdqnAction(int x, int y, int xt, int yt, float &totalReward);
    int ddpgAction(int x, int y, int xt, int yt, float &totalReward);
    int ppoAction(int x, int y, int xt, int yt, float &totalReward);
    int sacAction(int x, int y, int xt, int yt, float &totalReward);
    int supervisedAction(int x, int y, int xt, int yt, float &totalReward);
    void setTrain(bool on){trainFlag = on;}
private:
    bool simulateMove(int& x, int& y, int direct);
    void simulateMove(Snake &clone, int& x, int& y, int k);

};

#endif // AGENT_H
