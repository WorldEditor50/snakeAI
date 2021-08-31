#ifndef CONTROLLER_H
#define CONTROLLER_H
#include <QObject>
#include <vector>
#include "dqn.h"
#include "dpg.h"
#include "ddpg.h"
#include "ppo.h"
#include "common.h"
#include "snake.h"
using namespace std;
using namespace RL;
class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(QObject *parent);
    Controller(QObject *parent, vector<vector<int> >& map, Snake& s);
    ~Controller();
    void observe(vector<double>& statex, int x, int y, int xt, int yt, vector<double> &output);
    int astarAgent(int x, int y, int xt, int yt);
    int randAgent(int x, int y, int xt, int yt);
    int dqnAgent(int x, int y, int xt, int yt);
    int dpgAgent(int x, int y, int xt, int yt);
    int ddpgAgent(int x, int y, int xt, int yt);
    int ppoAgent(int x, int y, int xt, int yt);
    int supervisedAgent(int x, int y, int xt, int yt);
    double reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward3(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward4(int xi, int yi, int xn, int yn, int xt, int yt);
    bool move(int& x, int& y, int direct);
public:
    vector<vector<int> >& map;
    Snake &snake;
    vector<double> state;
    vector<double> nextState;
    DQN dqn;
    BPNN bpnn;
    DPG dpg;
    DDPG ddpg;
    PPO ppo;
signals:
    void totalReward(double r);
    void scale(int value);
public slots:
};

#endif // CONTROLLER_H
