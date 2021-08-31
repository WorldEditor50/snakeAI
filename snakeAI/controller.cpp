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
    this->bpnn = BPNN(stateDim, 16, 4, 4, 1);
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
    double x1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
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
    double x1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = 0;
    r = tanh(2 / (x1 - x2));
    return r;
}

double Controller::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
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
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = sqrt(x1) - sqrt(x2);

    return 1.5 * r / (1 + r * r);
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
        direct = RL::argmax(Action);
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
    observe(state, x, y, xt, yt, dqn.QMainNet.output());
    for (std::size_t j = 0; j < 256; j++) {
        int xi = xn;
        int yi = yn;
        Vec& action = dqn.greedyAction(state);
        int k = RL::argmax(action);
        move(xn, yn, k);
        r = reward4(xi, yi, xn, yn, xt, yt);
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
    emit totalReward(total);
    /* training */
    dqn.learn(OPT_RMSPROP, 8192, 1024, 64, 0.001);
    /* making decision */
    observe(state, x, y, xt, yt, dqn.QMainNet.output());
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
    observe(state, x, y, xt, yt, dpg.policyNet.output());
    for (std::size_t j = 0; j < 64; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = dpg.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        observe(nextState, xn, yn, xt, yt, dpg.policyNet.output());
        Step s;
        s.state = state;
        s.action  = dpg.policyNet.output();
        s.reward  = reward4(xi, yi, xn, yn, xt, yt);
        total += s.reward;
        steps.push_back(s);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    emit totalReward(total);
    /* training */
    dpg.reinforce(OPT_RMSPROP, 0.01, steps);
    /* making decision */
    observe(state, x, y, xt, yt, dpg.policyNet.output());
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
    double total = 0;
    observe(state, x, y, xt, yt, ddpg.actorP.output());
    for (std::size_t j = 0; j < 128; j++) {
        int xi = xn;
        int yi = yn;
        Vec & action = ddpg.greedyAction(state);
        int k = RL::argmax(action);
        move(xn, yn, k);
        r = reward1(xi, yi, xn, yn, xt, yt);
        total += r;
        observe(nextState, xn, yn, xt, yt, ddpg.actorP.output());
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
    emit totalReward(total);
    /* training */
    ddpg.learn(OPT_RMSPROP, 8192, 256, 64, 0.01, 0.02);
    /* making decision */
    observe(state, x, y, xt, yt, ddpg.actorP.output());
    a = ddpg.action(state);
    ddpg.actorP.show();
    return a;
}

int Controller::ppoAgent(int x, int y, int xt, int yt)
{
    int direct = 0;
    /* exploring environment */
    vector<Transition> trans;
    int xn = x;
    int yn = y;
    int s = 10;
    double total = 0;
    emit scale(s);
    observe(state, x, y, xt, yt, ppo.actorP.output());
    for (std::size_t j = 0; j < 16; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = ppo.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        observe(nextState, xn, yn, xt, yt, ppo.actorP.output());
        Transition t;
        t.state = state;
        t.action = ppo.actorQ.output();
        t.reward = reward4(xi, yi, xn, yn, xt, yt);
        total += t.reward;
        trans.push_back(t);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    emit totalReward(total * s);
    /* training */
    ppo.learnWithClipObject(OPT_RMSPROP, 0.01, trans);
    /* making decision */
    observe(state, x, y, xt, yt, ppo.actorP.output());
    direct = ppo.action(state);
    ppo.actorP.show();
    return direct;
}

int Controller::supervisedAgent(int x, int y, int xt, int yt)
{
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    bool training = true;
    double m = 0;
    vector<double>& action = bpnn.output();
    if (training) {
        observe(state, xn, yn, xt, yt, action);
        for (std::size_t i = 0; i < 128; i++) {
            bpnn.feedForward(state);
            direct1 = RL::argmax(action);
            direct2 = astarAgent(xn, yn, xt, yt);
            if (direct1 != direct2) {
                vector<double> target(4, 0);
                target[direct2] = 1;
                bpnn.gradient(state, target);
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
            bpnn.optimize(OPT_RMSPROP, 0.01);
        }
    }
    observe(state, x, y, xt, yt, action);
    bpnn.feedForward(state);
    direct1 = RL::argmax(action);
    bpnn.show();
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
