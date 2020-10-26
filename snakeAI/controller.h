#ifndef CONTROLLER_H
#define CONTROLLER_H
#include <QObject>
#include <vector>
#include "dqn.h"
#include "dpg.h"
#include "ddpg.h"
#include "ppo.h"
#include "common.h"

using namespace std;
using namespace ML;
class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(QObject *parent);
    Controller(QObject *parent, vector<vector<int> >& map);
    ~Controller();
    void setState(vector<double>& statex,int x, int y, int xt, int yt);
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
    bool move(int& x, int& y, int direct);
    int maxAction(vector<double>& Action);
    vector<vector<int> >& map;
    vector<double> state;
    vector<double> nextState;
    DQN dqn;
    MLP mlp;
    DPG dpg;
    DDPG ddpg;
    PPO ppo;
signals:
    void sigTotalReward(double r);
public slots:
};

#endif // CONTROLLER_H
