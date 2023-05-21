#ifndef AGENT_H
#define AGENT_H
#include <QObject>
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

using namespace RL;
class Agent : public QObject
{
    Q_OBJECT
public:
    explicit Agent(QObject *parent, Mat& map, Snake& s);
    ~Agent();
    void observe(Mat& statex, int x, int y, int xt, int yt);
    int acting(int x, int y, int xt, int yt);
    int astarAction(int x, int y, int xt, int yt);
    int randAction(int x, int y, int xt, int yt);
    int dqnAction(int x, int y, int xt, int yt);
    int qlstmAction(int x, int y, int xt, int yt);
    int dpgAction(int x, int y, int xt, int yt);
    int drpgAction(int x, int y, int xt, int yt);
    int ddpgAction(int x, int y, int xt, int yt);
    int ppoAction(int x, int y, int xt, int yt);
    int supervisedAction(int x, int y, int xt, int yt);
signals:
    void totalReward(float r);
    void scale(int value);
public slots:
    void setTrain(bool on){trainFlag = on;}
private:
    float reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward3(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward4(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward5(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward6(int xi, int yi, int xn, int yn, int xt, int yt);
    bool simulateMove(int& x, int& y, int direct);
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
    QLSTM qlstm;
    DRPG drpg;
    bool trainFlag;
};

#endif // AGENT_H
