#include "controller.h"

Controller::Controller(vector<vector<int> >& map):map(map)
{
    this->dqn.createNet(8, 16, 4, 4, 65532, 256, 64);
    this->dpg.createNet(8, 16, 4, 4, 0.1);
    this->ddpg.createNet(8, 16, 4, 4, 4096, 256, 64);
    this->bp.createNet(8, 16, 4, 4, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
    this->state.resize(8);
    this->nextState.resize(8);
}

double Controller::reward(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double x1 = (xi - xt)*(xi - xt) + (yi - yt) * (yi - yt);
    double x2 = (xn - xt)*(xn - xt) + (yn - yt) * (yn - yt);
    double r = tanh(x1 - x2);
    return r;
}

void Controller::setState(vector<double>& statex, int x, int y, int xt, int yt)
{
    double row = double(map.size());
    double col = double(map[0].size());
    statex[0] = double(x)/row;
    statex[1] = double(y)/col;
    statex[2] = double(xt)/row;
    statex[3] = double(yt)/col;
    statex[4] = statex[2] - statex[0];
    statex[5] = statex[3] - statex[1];
    statex[6] = statex[4] - statex[5];
    statex[7] = statex[4] + statex[5];
    return;
}

int Controller::AStarAgent(int x, int y, int xt, int yt)
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
    for (int i = 0; i < 4; i++) {
        if (minDistance > distance[i]) {
            minDistance = distance[i];
            minDirect = i;
        }
    }
    return minDirect;
}

int Controller::randomSearchAgent(int x, int y, int xt, int yt)
{
    int xn = x;
    int yn = y;
    int direct = 0;
    double gamma = 0.9;
    double T = 10000;
    vector<double> action(4, 0);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            move(xn, yn, direct);
            action[direct] = gamma * action[direct] + reward(xi, yi, xn, yn, xt, yt);
            if ((map[xn][yn] == 1) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal action */
        direct = maxAction(action);
        move(xn, yn, direct);
        if (map[xn][yn] != 1) {
            break;
        }
        /* punishment */
        action[direct] -= 2 * action[direct];
        T = 0.98 * T;
    }
    return direct;
}

int Controller::dqnAgent(int x, int y, int xt, int yt)
{
    int a = 0;
    int steps = 16;
    /* exploring environment */
    for (int i = 0; i < steps; i++) {
        int xn = x;
        int yn = y;
        double r = 0;
        setState(state, x, y, xt, yt);
        for (int j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            a = dqn.eGreedyAction(state);
            move(xn, yn, a);
            r = reward(xi, yi, xn, yn, xt, yt);
            setState(nextState, xn, yn, xt, yt);
            if (map[xn][yn] == 1) {
                dqn.perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                dqn.perceive(state, a, nextState, r, true);
                break;
            } else {
                dqn.perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
    }
    /* training */
    dqn.learn(OPT_RMSPROP, 0.01);
    /* making decision */
    setState(state, x, y, xt, yt);
    a = dqn.action(state);
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
    setState(state, x, y, xt, yt);
    for (int j = 0; j < 256; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        //direct = dpg.randomAction();
        direct = dpg.eGreedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
        Step s;
        s.state = state;
        s.action  = dpg.policyNet.getOutput();
        s.reward  = reward(xi, yi, xn, yn, xt, yt);
        steps.push_back(s);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    /* training */
    dpg.reinforce(steps);
    /* making decision */
    setState(state, x, y, xt, yt);
    direct = dpg.action(state);
    dpg.policyNet.show();
    return direct;
}

int Controller::ddpgAgent(int x, int y, int xt, int yt)
{
    int a = 0;
    int steps = 16;
    /* exploring environment */
    for (int i = 0; i < steps; i++) {
        int xn = x;
        int yn = y;
        double r = 0;
        setState(state, x, y, xt, yt);
        for (int j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            a = ddpg.eGreedyAction(state);
            //a = ddpg.randomAction();
            //a = ddpg.action(state);
            move(xn, yn, a);
            r = reward(xi, yi, xn, yn, xt, yt);
            setState(nextState, xn, yn, xt, yt);
            if (map[xn][yn] == 1) {
                ddpg.perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                ddpg.perceive(state, a, nextState, r, true);
                break;
            } else {
                ddpg.perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
    }
    /* training */
    ddpg.learn(OPT_RMSPROP, 0.01, 0.01);
    /* making decision */
    setState(state, x, y, xt, yt);
    a = ddpg.action(state);
    ddpg.ActorMainNet.show();
    return a;
}

int Controller::bpAgent(int x, int y, int xt, int yt)
{
    /* supervise learning */
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    bool training = true;
    double m = 0;
    vector<double>& action = bp.getOutput();
    if (training) {
        setState(state, x, y, xt, yt);
        for (int i = 0; i < 128; i++) {
            bp.feedForward(state);
            direct1 = maxAction(action);
            direct2 = AStarAgent(xn, yn, xt, yt);
            if (direct1 != direct2) {
                vector<double> target(4, 0);
                target[direct2] = 1;
                bp.calculateGradient(state, action, target);
                m++;
            }
            if ((xn == xt && yn == yt) || (map[xn][yn] == 1)) {
                break;
            }
            setState(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            bp.RMSProp(0.9, 0.01);
        }
    }
    setState(state, x, y, xt, yt);
    bp.feedForward(state);
    direct1 = maxAction(action);
    bp.show();
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
    for (int i = 0; i < action.size(); i++) {
        if (maxValue < action[i]) {
            maxValue = action[i];
            direct = i;
        }
    }
    return direct;
}
