#ifndef CONTROLLER_H
#define CONTROLLER_H
#include <vector>
#include "dqn.h"
#include "policyGradient.h"
#include "ddpg.h"
#include "ppo.h"
#include "common.h"
using namespace std;
using namespace ML;
class Controller
{
public:
    Controller(vector<vector<int> >& map);
    void setState(vector<double>& statex,int x, int y, int xt, int yt);
    int AStarAgent(int x, int y, int xt, int yt);
    int randomSearchAgent(int x, int y, int xt, int yt);
    int dqnAgent(int x, int y, int xt, int yt);
    int dpgAgent(int x, int y, int xt, int yt);
    int ddpgAgent(int x, int y, int xt, int yt);
    int ppoAgent(int x, int y, int xt, int yt);
    int bpAgent(int x, int y, int xt, int yt);
    double reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    bool move(int& x, int& y, int direct);
    int maxAction(vector<double>& Action);
    vector<vector<int> >& map;
    vector<double> state;
    vector<double> nextState;
    DQN dqn;
    BPNet bp;
    DPG dpg;
    DDPG ddpg;
    PPO ppo;
};

#endif // CONTROLLER_H
