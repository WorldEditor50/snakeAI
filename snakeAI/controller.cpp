#include "controller.h"

Controller::Controller(vector<vector<int> >& map):map(map)
{
    this->dqn.CreateNet(4, 32, 4, 4, 16384, 256, 64);
    this->dpg.CreateNet(4, 32, 4, 4, 0.1);
    this->ddpg.CreateNet(4, 32, 4, 4, 16384, 256, 64);
    this->ppo.CreateNet(4, 32, 4, 4, 1024, 256, 4);
    this->bp.CreateNet(4, 32, 4, 4, 1, ACTIVATE_SIGMOID, LOSS_MSE);
    this->state.resize(4);
    this->nextState.resize(4);
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
    double r = tanh(x1 - x2);
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
    double r = 0;
    if (x1 > x2) {
        r = 0.01;
    } else {
        r = -0.1;
    }
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
        Action[direct] -= 2 * Action[direct];
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
            a = dqn.GreedyAction(state);
            move(xn, yn, a);
            r = reward1(xi, yi, xn, yn, xt, yt);
            setState(nextState, xn, yn, xt, yt);
            if (map[xn][yn] == 1) {
                dqn.Perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                dqn.Perceive(state, a, nextState, r, true);
                break;
            } else {
                dqn.Perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
    }
    /* training */
    dqn.Learn(OPT_RMSPROP, 0.001);
    /* making decision */
    setState(state, x, y, xt, yt);
    a = dqn.Action(state);
    dqn.QMainNet.Show();
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
    for (int j = 0; j < 16; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = dpg.GreedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
        Step s;
        s.state = state;
        s.action  = dpg.policyNet.GetOutput();
        s.reward  = reward1(xi, yi, xn, yn, xt, yt);
        steps.push_back(s);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    /* training */
    dpg.Reinforce(steps, OPT_RMSPROP, 0.01);
    /* making decision */
    setState(state, x, y, xt, yt);
    direct = dpg.Action(state);
    dpg.policyNet.Show();
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
            a = ddpg.GreedyAction(state);
            move(xn, yn, a);
            r = reward2(xi, yi, xn, yn, xt, yt);
            setState(nextState, xn, yn, xt, yt);
            if (map[xn][yn] == 1) {
                ddpg.Perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                ddpg.Perceive(state, a, nextState, r, true);
                break;
            } else {
                ddpg.Perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
    }
    /* training */
    ddpg.Learn(OPT_RMSPROP, 0.001, 0.001);
    /* making decision */
    setState(state, x, y, xt, yt);
    a = ddpg.Action(state);
    ddpg.actorMainNet.Show();
    return a;
}

int Controller::ppoAgent(int x, int y, int xt, int yt)
{
    int direct = 0;
    /* exploring environment */
    vector<Step> steps;
    int xn = x;
    int yn = y;
    setState(state, x, y, xt, yt);
    for (int j = 0; j < 128; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = ppo.GreedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
        Step s;
        s.state = state;
        s.action  = ppo.actor.GetOutput();
        s.reward  = reward2(xi, yi, xn, yn, xt, yt);
        steps.push_back(s);
        if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
            break;
        }
        state = nextState;
    }
    /* training */
    ppo.Perceive(steps);
    ppo.Learn(OPT_RMSPROP, 0.01);
    /* making decision */
    setState(state, x, y, xt, yt);
    direct = ppo.Action(state);
    ppo.actor.Show();
    return direct;
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
    vector<double>& action = bp.GetOutput();
    if (training) {
        setState(state, xn, yn, xt, yt);
        for (int i = 0; i < 128; i++) {
            bp.FeedForward(state);
            direct1 = maxAction(action);
            direct2 = AStarAgent(xn, yn, xt, yt);
            if (direct1 != direct2) {
                vector<double> target(4, 0);
                target[direct2] = 1;
                bp.Gradient(state, target);
                m++;
            }
            if ((xn == xt) && (yn == yt)) {
                break;
            }
            if (map[xn][yn] == 1) {
                break;
            }
            setState(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            bp.Optimize(OPT_RMSPROP, 0.01);
        }
    }
    setState(state, x, y, xt, yt);
    bp.FeedForward(state);
    direct1 = maxAction(action);
    bp.Show();
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
