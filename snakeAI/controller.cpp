#include "controller.h"
#include <QDebug>
Controller::Controller(QObject *parent, vector<vector<int> >& map, Snake &s):
    QObject(parent),
    map(map),
    snake(s)
{
    int stateDim = 8;
    this->dqn = DQN(stateDim, 16, 4, 4);
    this->dpg = DPG(stateDim, 16, 4, 4);
    this->ddpg = DDPG(stateDim, 16, 4, 4);
    this->ppo = PPO(stateDim, 16, 4, 4);
    this->mlp = MLP(stateDim, 16, 4, 4, 1);
    this->state.resize(stateDim);
    this->nextState.resize(stateDim);
}

Controller::~Controller()
{

}

double Controller::reward1(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt)*(xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt)*(xn - xt) + (yn - yt) * (yn - yt);
    double r = sqrt(x1) - sqrt(x2);
    r = r / (sqrt(r * r + 1));
    return r;
}

double Controller::reward2(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt)*(xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt)*(xn - xt) + (yn - yt) * (yn - yt);
    return tanh(1 / (x1 - x2));
}

double Controller::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt)*(xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt)*(xn - xt) + (yn - yt) * (yn - yt);
    double r = 0;
    if (x1 > x2) {
        r = 0.1;
    } else {
        r = -0.1;
    }
    return r;
}

double Controller::reward4(int xi, int yi, int xn, int yn, int xt, int yt)
{
    double di = fabs(double(xi - xt)) + fabs(double(yi - yt));
    double dn = fabs(double(xn - xt)) + fabs(double(yn - yt));
    double r = di - dn;
    if (map[xn][yn] == 1) {
        return -fabs(r) * 10;
    }
    if (xn == xt && yn == yt) {
        return fabs(r) * 10;
    }
    return r;
}

void Controller::observe(vector<double>& statex, int x, int y, int xt, int yt, vector<double>& output)
{
    double xc = double(map.size()) / 2;
    double yc = double(map[0].size()) / 2;
    statex[0] = (x - xc) / xc;
    statex[1] = (y - yc) / yc;
    statex[2] = (xt - xc) / xc;
    statex[3] = (yt - yc) / yc;
    statex[4] = output[0];
    statex[5] = output[1];
    statex[6] = output[2];
    statex[7] = output[3];
    return;
}

int Controller::astarAgent(int x, int y, int xt, int yt)
{
    int distance[4];
    for(int i = 0; i < 4; i++) {
        int agenxt = x;
        int agenyt = y;
        if(move(agenxt, agenyt, i)) {
            distance[i] = (agenxt - xt) * (agenxt - xt) + (agenyt - yt) * (agenyt - yt);
        } else {
            distance[i] = 10000;
        }
    }
    int minDirect = 0;
    int minDistance = distance[0];
    for (std::size_t i = 0; i < 4; i++) {
        if (minDistance > distance[i]) {
            minDistance = distance[i];
            minDirect = i;
        }
    }
    return minDirect;
}

int Controller::randAgent(int x, int y, int xt, int yt)
{
    int xn = x;
    int yn = y;
    int direct = 0;
    double gamma = 0.9;
    double T = 10000;
    vector<double> Action(4, 0);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            move(xn, yn, direct);
            Action[direct] = gamma * Action[direct] + reward1(xi, yi, xn, yn, xt, yt);
            if ((map[xn][yn] == 1) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal Action */
        direct = maxAction(Action);
        move(xn, yn, direct);
        if (map[xn][yn] != 1) {
            break;
        }
        /* punishment */
        Action[direct] *= -2;
        T = 0.98 * T;
    }
    return direct;
}

int Controller::dqnAgent(int x, int y, int xt, int yt)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    double r = 0;
    double total = 0;
    observe(state, x, y, xt, yt, dqn.QMainNet.getOutput());
    for (std::size_t j = 0; j < 256; j++) {
        int xi = xn;
        int yi = yn;
        Vec& action = dqn.greedyAction(state);
        int k = maxAction(action);
        move(xn, yn, k);
        r = reward1(xi, yi, xn, yn, xt, yt);
        //std::cout<<r<<std::endl;
        total += r;
        observe(nextState, xn, yn, xt, yt, action);
        if (map[xn][yn] == 1) {
            dqn.perceive(state, action, nextState, r, true);
            break;
        }
        if (xn == xt && yn == yt) {
            dqn.perceive(state, action, nextState, r, true);
            break;
        } else {
            dqn.perceive(state, action, nextState, r, false);
        }
        state = nextState;
    }
    emit sigTotalReward(total);
    /* training */
    dqn.learn(OPT_RMSPROP, 8192, 1024, 64, 0.001);
    /* making decision */
    observe(state, x, y, xt, yt, dqn.QMainNet.getOutput());
    int a = dqn.action(state);
    dqn.QMainNet.show();
    return a;
}

int Controller::dpgAgent(int x, int y, int xt, int yt)
{
    int direct = 0;
    /* exploring environment */
    vector<Step> steps;
    int xn = x;
    int yn = y;
    double total = 0;
    observe(state, x, y, xt, yt, dpg.policyNet.getOutput());
    for (std::size_t j = 0; j < 64; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = dpg.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        observe(nextState, xn, yn, xt, yt, dpg.policyNet.getOutput());
        Step s;
        s.state = state;
        s.action  = dpg.policyNet.getOutput();
        s.reward  = reward1(xi, yi, xn, yn, xt, yt);
        total += s.reward;
        steps.push_back(s);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    emit sigTotalReward(total);
    /* training */
    dpg.reinforce(OPT_RMSPROP, 0.01, steps);
    /* making decision */
    observe(state, x, y, xt, yt, dpg.policyNet.getOutput());
    direct = dpg.action(state);
    dpg.policyNet.show();
    return direct;
}

int Controller::ddpgAgent(int x, int y, int xt, int yt)
{
    int a = 0;
    /* exploring environment */
    int xn = x;
    int yn = y;
    double r = 0;
    double totalReward = 0;
    observe(state, x, y, xt, yt, ddpg.actorP.getOutput());
    for (std::size_t j = 0; j < 128; j++) {
        int xi = xn;
        int yi = yn;
        Vec & action = ddpg.greedyAction(state);
        int k = maxAction(action);
        move(xn, yn, k);
        r = reward1(xi, yi, xn, yn, xt, yt);
        totalReward += r;
        observe(nextState, xn, yn, xt, yt, ddpg.actorP.getOutput());
        if (map[xn][yn] == 1) {
            ddpg.perceive(state, action, nextState, r, true);
            break;
        }
        if (xn == xt && yn == yt) {
            ddpg.perceive(state, action, nextState, r, true);
            break;
        } else {
            ddpg.perceive(state, action, nextState, r, false);
        }
        state = nextState;
    }
    emit sigTotalReward(totalReward);
    /* training */
    ddpg.learn(OPT_RMSPROP, 8192, 256, 64, 0.01, 0.02);
    /* making decision */
    observe(state, x, y, xt, yt, ddpg.actorP.getOutput());
    a = ddpg.action(state);
    ddpg.actorP.show();
    return a;
}

int Controller::ppoAgent(int x, int y, int xt, int yt)
{
    int direct = 0;
    /* exploring environment */
    vector<Transit> trans;
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt, ppo.actorP.getOutput());
    double total = 0;
    for (std::size_t j = 0; j < 64; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = ppo.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        observe(nextState, xn, yn, xt, yt, ppo.actorP.getOutput());
        Transit t;
        t.state = state;
        t.action = ppo.actorQ.getOutput();
        t.reward = reward1(xi, yi, xn, yn, xt, yt);
        total += t.reward;
        trans.push_back(t);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    emit sigTotalReward(total);
    /* training */
    ppo.learnWithClipObject(OPT_RMSPROP, 0.01, trans);
    /* making decision */
    observe(state, x, y, xt, yt, ppo.actorP.getOutput());
    direct = ppo.action(state);
    ppo.actorP.show();
    return direct;
}

int Controller::supervisedAgent(int x, int y, int xt, int yt)
{
    /*   learning */
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    bool training = true;
    double m = 0;
    vector<double>& action = mlp.getOutput();
    if (training) {
        observe(state, xn, yn, xt, yt, action);
        for (std::size_t i = 0; i < 128; i++) {
            mlp.feedForward(state);
            direct1 = maxAction(action);
            direct2 = astarAgent(xn, yn, xt, yt);
            if (direct1 != direct2) {
                vector<double> target(4, 0);
                target[direct2] = 1;
                mlp.gradient(state, target);
                m++;
            }
            if ((xn == xt) && (yn == yt)) {
                break;
            }
            if (map[xn][yn] == 1) {
                break;
            }
            observe(state, xn, yn, xt, yt, action);
        }
        if (m > 0) {
            mlp.optimize(OPT_RMSPROP, 0.01);
        }
    }
    observe(state, x, y, xt, yt, action);
    mlp.feedForward(state);
    direct1 = maxAction(action);
    mlp.show();
    return direct1;
}

bool Controller::move(int& x, int& y, int direct)
{
    moving(x, y, direct);
    bool flag = true;
    if (map[x][y] == 1) {
        flag = false;
    }
    return flag;
}

int Controller::maxAction(vector<double>& action)
{
    double maxValue = action[0];
    int direct = 0;
    for (std::size_t i = 0; i < action.size(); i++) {
        if (maxValue < action[i]) {
            maxValue = action[i];
            direct = i;
        }
    }
    return direct;
}
