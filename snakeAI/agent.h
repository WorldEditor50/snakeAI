#ifndef AGENT_H
#define AGENT_H
#include <vector>
#include "dqn.h"
#include "dpg.h"
#include "ddpg.h"
#include "ppo.h"
#include "qlstm.h"
#include "drpg.h"
#include "common.h"
#include "snake.h"
#include "mat.hpp"
#include "sac.h"

using namespace RL;
class Agent
{
private:
    Mat& map;
    Snake &snake;
    Mat state;
    Mat nextState;
    DQN dqn;
    BPNN bpnn;
    DPG dpg;
    DDPG ddpg;
    PPO ppo;
    SAC sac;
    QLSTM qlstm;
    DRPG drpg;
    bool trainFlag;
public:
    explicit Agent(Mat& map, Snake& s);
    ~Agent();
    void observe(Mat& statex, int x, int y, int xt, int yt);
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
