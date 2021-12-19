#include "agent.h"
#include <QDebug>
Agent::Agent(QObject *parent, vector<vector<int> >& map, Snake &s):
    QObject(parent),
    map(map),
    snake(s),
    trainFlag(true)
{
    int stateDim = 4;
    dqn = DQN(stateDim, 16, 4);
    dpg = DPG(stateDim, 16, 4);
    ddpg = DDPG(stateDim, 16, 4);
    ppo = PPO(stateDim, 16, 4);
    bpnn = BPNN(BPNN::Layers{
                          Layer<Sigmoid>::_(stateDim, 16, true),
                          Layer<Sigmoid>::_(16, 16, true),
                          Layer<Sigmoid>::_(16, 16, true),
                          Layer<Sigmoid>::_(16, 4, true)
                      });
    qlstm = QLSTM(stateDim, 16, 4);
    state.resize(stateDim);
    nextState.resize(stateDim);
    dqn.load("./dqn");
    dpg.load("./dpg");
    ddpg.load("./ddpg_actor", "./ddpg_critic");
    bpnn.load("./bpnn");
    ppo.load("./ppo_actor", "./ppo_critic");
}

Agent::~Agent()
{
    dqn.save("./dqn");
    dpg.save("./dpg");
    ddpg.save("./ddpg_actor", "./ddpg_critic");
    bpnn.save("./bpnn");
    ppo.save("./ppo_actor", "./ppo_critic");
}

double Agent::reward1(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = sqrt(d1) - sqrt(d2);
    return r / sqrt(r * r + 1);
}

double Agent::reward2(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = d1 - d2;
    return r / sqrt(1 + r*r);
}

double Agent::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = 0;
    if (d1 > d2) {
        r = 0.8;
    } else {
        r = -0.8;
    }
    return r;
}

double Agent::reward4(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    double r = sqrt(d1) - sqrt(d2);
    return 1.5 * r / (1 + r * r);
}

double Agent::reward5(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    double d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    double d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return sqrt(d1) - sqrt(d2);
}

double Agent::reward6(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map[xn][yn] == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    return 0;
}

void Agent::observe(vector<double>& statex, int x, int y, int xt, int yt)
{
    double xc = double(map.size()) / 2;
    double yc = double(map[0].size()) / 2;
    statex[0] = (x - xc) / xc;
    statex[1] = (y - yc) / yc;
    statex[2] = (xt - xc) / xc;
    statex[3] = (yt - yc) / yc;
    return;
}

int Agent::acting(int x, int y, int xt, int yt)
{
    return dqnAction(x, y, xt, yt);
}

int Agent::astarAction(int x, int y, int xt, int yt)
{
    int distance[4];
    for(int i = 0; i < 4; i++) {
        int agentxt = x;
        int agentyt = y;
        if(simulateMove(agentxt, agentyt, i)) {
            distance[i] = (agentxt - xt) * (agentxt - xt) + (agentyt - yt) * (agentyt - yt);
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

int Agent::randAction(int x, int y, int xt, int yt)
{
    int xn = x;
    int yn = y;
    int direct = 0;
    double gamma = 0.9;
    double T = 10000;
    std::vector<double> act(4, 0);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            simulateMove(xn, yn, direct);
            act[direct] = gamma * act[direct] + reward4(xi, yi, xn, yn, xt, yt);
            if ((map[xn][yn] == 1) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal Action */
        direct = RL::argmax(act);
        simulateMove(xn, yn, direct);
        if (map[xn][yn] != 1) {
            break;
        }
        /* punishment */
        act[direct] *= -2;
        T = 0.98 * T;
    }
    return direct;
}

int Agent::dqnAction(int x, int y, int xt, int yt)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    double r = 0;
    double total = 0;
    observe(state, x, y, xt, yt);
    std::vector<double> state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            Vec& action = dqn.sample(state);
            int k = RL::argmax(action);
            simulateMove(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
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
        dqn.learn(OPT_RMSPROP, 8192, 256, 64, 0.001);
    }
    /* making decision */
    return dqn.action(state_);
}

int Agent::qlstmAction(int x, int y, int xt, int yt)
{
    int xn = x;
    int yn = y;
    double r = 0;
    double total = 0;
    observe(state, x, y, xt, yt);
    std::vector<double> state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 128; j++) {
            int xi = xn;
            int yi = yn;
            Vec& action = qlstm.sample(state);
            int k = RL::argmax(action);
            simulateMove(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map[xn][yn] == 1) {
                qlstm.perceive(state, action, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                qlstm.perceive(state, action, nextState, r, true);
                break;
            } else {
                qlstm.perceive(state, action, nextState, r, false);
            }
            state = nextState;
        }
        emit totalReward(total);
        /* training */
        qlstm.learn(8192, 256, 32, 0.001);
    }
    /* making decision */
    Vec &action = qlstm.action(state_);
    for (std::size_t i = 0; i < action.size(); i++) {
        std::cout<<action[i]<<" ";
    }
    std::cout<<std::endl;
    return RL::argmax(action);
}

int Agent::dpgAction(int x, int y, int xt, int yt)
{
    int direct = 0;
    /* exploring environment */
    std::vector<Step> steps;
    int xn = x;
    int yn = y;
    double total = 0;
    observe(state, x, y, xt, yt);
    std::vector<double> state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 16; j++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            Vec &output = dpg.sample(state);
            direct = RL::argmax(output);
            simulateMove(xn, yn, direct);
            observe(nextState, xn, yn, xt, yt);
            Step s;
            s.state = state;
            s.action  = output;
            s.reward  = reward1(xi, yi, xn, yn, xt, yt);
            total += s.reward;
            steps.push_back(s);
            if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        emit totalReward(total);
        /* training */
        dpg.reinforce(OPT_RMSPROP, 0.001, steps);
    }
    /* making decision */
    direct = dpg.action(state_);
    return direct;
}

int Agent::ddpgAction(int x, int y, int xt, int yt)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    double r = 0;
    double total = 0;
    observe(state, x, y, xt, yt);
    vector<double> state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 32; j++) {
            int xi = xn;
            int yi = yn;
            Vec & action = ddpg.sample(state);
            int k = RL::argmax(action);
            simulateMove(xn, yn, k);
            r = reward4(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
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
        ddpg.learn(OPT_RMSPROP, 8192, 256, 32, 0.001, 0.0001);
    }
    return ddpg.action(state_);
}

int Agent::ppoAction(int x, int y, int xt, int yt)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    double total = 0;
    observe(state, x, y, xt, yt);
    std::vector<double> state_ = state;
    if (trainFlag == true) {
        std::vector<Transition> trans;
        for (std::size_t j = 0; j < 32; j++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            Vec &output = ppo.sample(state);
            int direct = RL::argmax(output);
            /* move */
            simulateMove(xn, yn, direct);
            observe(nextState, xn, yn, xt, yt);
            Transition transition;
            transition.state = state;
            transition.action = output;
            transition.reward = reward4(xi, yi, xn, yn, xt, yt);
            total += transition.reward;
            trans.push_back(transition);
            if (map[xn][yn] == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        emit totalReward(total);
        /* training */
        ppo.learnWithClipObject(OPT_RMSPROP, 0.01, trans);
    }
    /* making decision */
    auto &p = ppo.action(state_);
    p.show();
    return p.argmax();
}

int Agent::supervisedAction(int x, int y, int xt, int yt)
{
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    double m = 0;
    std::vector<double>& action = bpnn.output();
    if (trainFlag == true) {
        observe(state, xn, yn, xt, yt);
        for (std::size_t i = 0; i < 128; i++) {
            bpnn.feedForward(state);
            direct1 = RL::argmax(action);
            direct2 = astarAction(xn, yn, xt, yt);
            if (direct1 != direct2) {
                std::vector<double> target(4, 0);
                target[direct2] = 1;
                bpnn.gradient(state, target, Loss::MSE);
                m++;
            }
            if ((xn == xt) && (yn == yt)) {
                break;
            }
            if (map[xn][yn] == 1) {
                break;
            }
            observe(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            bpnn.optimize(OPT_RMSPROP, 0.01);
        }
    }
    observe(state, x, y, xt, yt);
    bpnn.feedForward(state);
    direct1 = RL::argmax(action);
    bpnn.show();
    return direct1;
}

bool Agent::simulateMove(int& x, int& y, int direct)
{
    moving(x, y, direct);
    bool flag = true;
    if (map[x][y] == 1) {
        flag = false;
    }
    return flag;
}
