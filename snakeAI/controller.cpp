#include "controller.h"

Controller::Controller(QObject *parent, vector<vector<int> >& map):
    QObject(parent),
    map(map)
{
    this->dqn = DQN(8, 16, 4, 4);
    this->dpg = DPG(8, 16, 4, 4);
    this->ddpg = DDPG(8, 16, 4, 4);
    this->ppo = PPO(8, 16, 4, 4);
    this->mlp = MLP(8, 16, 4, 4, 1, ACTIVATE_SIGMOID, LOSS_MSE);
    this->state.resize(8);
    this->nextState.resize(8);
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
    double x1 = sqrt((xi - xt)*(xi - xt) + (yi - yt) * (yi - yt));
    double x2 = sqrt((xn - xt)*(xn - xt) + (yn - yt) * (yn - yt));
    double r = 0;
    if (x1 > x2) {
        r = tanh(1 / (x1 - x2));
    } else {
        r = tanh(x1 - x2);
    }
    return r;
}

double Controller::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    double row = double(map.size());
    double col = double(map[0].size());
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
    statex[4] = statex[2] - statex[0];
    statex[5] = statex[3] - statex[1];
    statex[6] = statex[4] - statex[5];
    statex[7] = statex[4] + statex[5];
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
    for (int i = 0; i < 4; i++) {
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
    int steps = 1;
    /* exploring environment */
    for (int i = 0; i < steps; i++) {
        int xn = x;
        int yn = y;
        double r = 0;
        double total = 0;
        setState(state, x, y, xt, yt);
        for (int j = 0; j < 512; j++) {
            int xi = xn;
            int yi = yn;
            std::vector<double>& action = dqn.greedyAction(state);
            int k = maxAction(action);
            move(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            total += r;
            setState(nextState, xn, yn, xt, yt);
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
    }
    /* training */
    dqn.learn(OPT_RMSPROP, 8192, 1024, 64, 0.0001);
    /* making decision */
    setState(state, x, y, xt, yt);
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
    setState(state, x, y, xt, yt);
    for (int j = 0; j < 256; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = dpg.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
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
    setState(state, x, y, xt, yt);
    direct = dpg.action(state);
    dpg.policyNet.show();
    return direct;
}

int Controller::ddpgAgent(int x, int y, int xt, int yt)
{
    int a = 0;
    int steps = 1;
    /* exploring environment */
    for (int i = 0; i < steps; i++) {
        int xn = x;
        int yn = y;
        double r = 0;
        double totalReward = 0;
        setState(state, x, y, xt, yt);
        for (int j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            std::vector<double> & action = ddpg.greedyAction(state);
            int k = maxAction(action);
            move(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            totalReward += r;
            setState(nextState, xn, yn, xt, yt);
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
    }
    /* training */
    ddpg.learn(OPT_RMSPROP, 8192, 256, 64, 0.01, 0.02);
    /* making decision */
    setState(state, x, y, xt, yt);
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
    setState(state, x, y, xt, yt);
    double total = 0;
    for (int j = 0; j < 128; j++) {
        int xi = xn;
        int yi = yn;
        /* Monte Carlo method */
        direct = ppo.greedyAction(state);
        /* Markov Chain */
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
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
    ppo.learnWithKLpenalty(OPT_RMSPROP, 0.01, trans);
    /* making decision */
    setState(state, x, y, xt, yt);
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
        setState(state, xn, yn, xt, yt);
        for (int i = 0; i < 128; i++) {
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
            setState(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            mlp.optimize(OPT_RMSPROP, 0.01);
        }
    }
    setState(state, x, y, xt, yt);
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
    for (int i = 0; i < action.size(); i++) {
        if (maxValue < action[i]) {
            maxValue = action[i];
            direct = i;
        }
    }
    return direct;
}
