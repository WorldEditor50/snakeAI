#include "controller.h"

Controller::Controller(vector<vector<int> >& map):map(map)
{
    this->dqn.createNet(8, 16, 4, 4, 65532, 256, 128, 0.0001);
    this->bp.createNetWithSoftmax(8, 16, 4, 4, ACTIVATE_SIGMOID);
    this->dpg.createNet(8, 16, 4, 4, 0.01);
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
    double r = 0;
    if (x1 > x2) {
        r = 1 / (x1 - x2);
    } else {
        r = (x1 - x2) / 100;
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
#if 0
    double maxVal = statex[0];
    double minVal = statex[0];
    for (int i = 0; i < statex.size(); i++) {
        if (statex[i] > maxVal) {
            maxVal = statex[i];
        }
        if (statex[i] < minVal) {
            minVal = statex[i];
        }
    }
    for (int i = 0; i < statex.size(); i++) {
        statex[i] = (statex[i] - minVal) / (maxVal - minVal);
    }
#endif
    //cout<<statex[0]<<" "<<statex[1]<<" "<<statex[2]<<" "<<statex[3]<<" "<<statex[4]<<" "<<statex[5]<<" "<<statex[6]<<" "<<statex[7]<<endl;
    return;
}

int Controller::simpleAgent(int x, int y, int xt, int yt)
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

int Controller::simpleAgent2(vector<Point>& body, int xt, int yt)
{
    int distance[4];
    int agentx = 0;
    int agenty = 0;
    for(int i = 0; i < 4; i++) {
        agentx = body[0].x;
        agenty = body[0].y;
        if (move(agentx, agenty, i)) {
           distance[i] = abs(agentx - xt) + abs(agenty - yt);
        } else {
            distance[i] = 20000;
        }

        for (int j = 1; j < body.size(); j++) {
            if (agentx == body[j].x && agenty == body[j].y) {
                distance[i] = 10000;
            }
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
            action[direct] = action[direct] * gamma + reward(xi, yi, xn, yn, xt, yt);
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
        /* punish */
        action[direct] -= 2 * action[direct];
        T = 0.98 * T;
    }
    return direct;
}

int Controller::dqnAgent(int x, int y, int xt, int yt)
{
    int direct = 0;
    int steps = 16;
    double r = 1;
    /* exploring environment */
    vector<Transition> transitions;
    for (int i = 0; i < steps; i++) {
        int xn = x;
        int yn = y;
        setState(state, x, y, xt, yt);
        for (int j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            direct = dqn.eGreedyAction(state);
            move(xn, yn, direct);
            r = reward(xi, yi, xn, yn, xt, yt);
            vector<double>& action = dqn.QMainNet.getOutput();
            setState(nextState, xn, yn, xt, yt);
            Transition trans;
            trans.state = state;
            trans.nextState = nextState;
            trans.action = action;
            trans.reward = r;
            if (map[xn][yn] == 1) {
                //dqn.perceive(state, action, nextState, r, true);
                trans.done = true;
                break;
            }
            if (xn == xt && yn == yt) {
                //dqn.perceive(state, action, nextState, r, true);
                trans.done = true;
                break;
            } else {
                trans.done = true;
                //dqn.perceive(state, action, nextState, r, false);
            }
            state = nextState;
            transitions.push_back(trans);
        }
    }
    /* training */
    //dqn.learn();
    dqn.onlineLearning(transitions);
    /* making decision */
    setState(state, x, y, xt, yt);
    direct = dqn.action(state);
    vector<double>& action = dqn.QMainNet.getOutput();
    std::cout<<action[0]<<" "<<action[1]<<" "<<action[2]<<" "<<action[3]<<std::endl;
    return direct;
}

int Controller::dpgAgent(int x, int y, int xt, int yt)
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
        direct = dpg.eGreedyAction(state);
        move(xn, yn, direct);
        setState(nextState, xn, yn, xt, yt);
        Step s;
        s.state = state;
        s.reward  = reward(xi, yi, xn, yn, xt, yt);
        s.action  = dpg.policyNet.getOutput();
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
    vector<double>& action = dpg.policyNet.getOutput();
    std::cout<<action[0]<<" "<<action[1]<<" "<<action[2]<<" "<<action[3]<<std::endl;
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
    vector<double>& action = bp.getOutput();
    if (training) {
        setState(state, x, y, xt, yt);
        for (int i = 0; i < 128; i++) {
            bp.feedForward(state);
            direct1 = maxAction(action);
            direct2 = simpleAgent(xn, yn, xt, yt);
            if (direct1 != direct2) {
                vector<double> target(4, 0);
                target[direct2] = 1;
                bp.calculateBatchGradient(state, action, target);
                m++;
            }
            if (xn == xt && yn == yt || map[xn][yn] == 1) {
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
    std::cout<<action[0]<<" "<<action[1]<<" "<<action[2]<<" "<<action[3]<<std::endl;
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
int Controller::MonteCarloAgent(vector<Point>& body, int xt, int yt)
{
    int x = body[0].x;
    int y = body[0].y;
    int xn = x;
    int yn = y;
    int direct = 0;
    double T = 10000;
    vector<Point> bodyClone1(body);
    vector<Point> bodyClone2(body);
    vector<double> action(4, 0);
    while (T > 0.001) {
        double gamma = 1;
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = bodyClone1[0].x;
            int yi = bodyClone1[0].y;
            trymove(bodyClone1, direct);
            action[direct] += reward(xi, yi, bodyClone1[0].x, bodyClone1[0].y, xt, yt, bodyClone1) * gamma;
            if ((xn == xt && yn == yt) || check(bodyClone1) == false) {
                break;
            }
            gamma = gamma * 0.9;
        }
        direct = maxAction(action);
        trymove(bodyClone2, direct);
        if (check(bodyClone2)) {
            break;
        }
        action[direct] -= 4 * action[direct];
        T = 0.98 * T;
    }
    return direct;
}
double Controller::reward(int xi, int yi, int xn, int yn, int xt, int yt, vector<Point>& body)
{
    if (xn == xt && yn == yt) {
        return 1;
    }
    if (check(body) == false) {
        return -1;
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

bool Controller::check(vector<Point> &body)
{
    for (int i = 1; i < body.size(); i++) {
        if (body[0].x == body[i].x && body[0].y == body[i].y) {
            return false;
        }
    }
    if (map[body[0].x][body[0].y] == 1) {
        return false;
    }
    return true;
}

void Controller::trymove(vector<Point>& body, int direct)
{
    for (int i = body.size() - 1; i > 0; i--) {
        body[i].x = body[i - 1].x;
        body[i].y = body[i - 1].y;
    }
    moving(body[0].x, body[0].y, direct);
    return;
}
