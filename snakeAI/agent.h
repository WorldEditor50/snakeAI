#ifndef AGENT_H
#define AGENT_H
#include <vector>
#include "rl/dqn.h"
#include "rl/dpg.h"
#include "rl/ddpg.h"
#include "rl/ppo.h"
#include "rl/qlstm.h"
#include "rl/drpg.h"
#include "rl/mat.hpp"
#include "rl/sac.h"
#include "common.h"
#include "snake.h"

class Agent
{
private:
    RL::Mat& map;
    Snake &snake;
    RL::Mat state;
    RL::Mat nextState;
    RL::DQN dqn;
    RL::BPNN bpnn;
    RL::DPG dpg;
    RL::DDPG ddpg;
    RL::PPO ppo;
    RL::SAC sac;
    RL::QLSTM qlstm;
    RL::DRPG drpg;
    bool trainFlag;
public:
    explicit Agent(RL::Mat& map, Snake& s);
    ~Agent();
    void observe(RL::Mat& statex, int x, int y, int xt, int yt);
    int astarAction(int x, int y, int xt, int yt, float &totalReward);
    int randAction(int x, int y, int xt, int yt, float &totalReward);
    int dqnAction(int x, int y, int xt, int yt, float &totalReward);
    int qlstmAction(int x, int y, int xt, int yt, float &totalReward);
    int dpgAction(int x, int y, int xt, int yt, float &totalReward);
    int drpgAction(int x, int y, int xt, int yt, float &totalReward);
    int ddpgAction(int x, int y, int xt, int yt, float &totalReward);
    int ppoAction(int x, int y, int xt, int yt, float &totalReward);
    int sacAction(int x, int y, int xt, int yt, float &totalReward);
    int supervisedAction(int x, int y, int xt, int yt, float &totalReward);
    void setTrain(bool on){trainFlag = on;}
private:
    float reward0(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward3(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward4(int xi, int yi, int xn, int yn, int xt, int yt);
    bool simulateMove(int& x, int& y, int direct);

};

#endif // AGENT_H
