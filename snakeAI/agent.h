#ifndef AGENT_H
#define AGENT_H
#include <QObject>
#include <vector>
#include "dqn.h"
#include "dpg.h"
#include "ddpg.h"
#include "ppo.h"
#include "qlstm.h"
#include "common.h"
#include "snake.h"

using namespace RL;
class Agent : public QObject
{
    Q_OBJECT
public:
    explicit Agent(QObject *parent, std::vector<std::vector<int> >& map, Snake& s);
    ~Agent();
    void observe(std::vector<double>& statex, int x, int y, int xt, int yt);
    int acting(int x, int y, int xt, int yt);
    int astarAction(int x, int y, int xt, int yt);
    int randAction(int x, int y, int xt, int yt);
    int dqnAction(int x, int y, int xt, int yt);
    int qlstmAction(int x, int y, int xt, int yt);
    int dpgAction(int x, int y, int xt, int yt);
    int ddpgAction(int x, int y, int xt, int yt);
    int ppoAction(int x, int y, int xt, int yt);
    int supervisedAction(int x, int y, int xt, int yt);
signals:
    void totalReward(double r);
    void scale(int value);
public slots:
    void setTrain(bool on){trainFlag = on;}
private:
    double reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward3(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward4(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward5(int xi, int yi, int xn, int yn, int xt, int yt);
    double reward6(int xi, int yi, int xn, int yn, int xt, int yt);
    bool simulateMove(int& x, int& y, int direct);
private:
    std::vector<std::vector<int> >& map;
    Snake &snake;
    std::vector<double> state;
    std::vector<double> nextState;
    DQN dqn;
    BPNN bpnn;
    DPG dpg;
    DDPG ddpg;
    PPO ppo;
    QLSTM qlstm;
    bool trainFlag;
};

#endif // AGENT_H
