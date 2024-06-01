#include "agent.h"
#include <QDebug>
Agent::Agent(RL::Mat& map, Snake &s):
    map(map),
    snake(s),
    trainFlag(true)
{
    int stateDim = 4;
    dqn = RL::DQN(stateDim, 16, 4);
    dpg = RL::DPG(stateDim, 16, 4);
    ddpg = RL::DDPG(stateDim, 16, 4);
    ppo = RL::PPO(stateDim, 16, 4);
    sac = RL::SAC(stateDim, 16, 4);
    bpnn = RL::BPNN(RL::Layer<RL::Sigmoid>::_(stateDim, 16, true),
                    RL::Layer<RL::Sigmoid>::_(16, 16, true),
                    RL::Layer<RL::Sigmoid>::_(16, 16, true),
                    RL::Layer<RL::Sigmoid>::_(16, 4, true));
    qlstm = RL::QLSTM(stateDim, 16, 4);
    drpg = RL::DRPG(stateDim, 16, 4);
    state = RL::Mat(stateDim, 1);
    nextState = RL::Mat(stateDim, 1);
#if 0
    dqn.load("./dqn");
    dpg.load("./dpg");
    ddpg.load("./ddpg_actor", "./ddpg_critic");
    bpnn.load("./bpnn");
    ppo.load("./ppo_actor", "./ppo_critic");
#endif
}

Agent::~Agent()
{
    dqn.save("./dqn");
    dpg.save("./dpg");
    ddpg.save("./ddpg_actor", "./ddpg_critic");
    bpnn.save("./bpnn");
    ppo.save("./ppo_actor", "./ppo_critic");
}

float Agent::reward0(int xi, int yi, int xn, int yn, int xt, int yt)
{
    /* agent goes out of the map */
    if (map(xn, yn) == 1) {
        return -1;
    }
    /* agent reaches to the target's position */
    if (xn == xt && yn == yt) {
        return 1;
    }
    /* the distance from agent's previous position to the target's position */
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    /* the distance from agent's current position to the target's position */
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return std::sqrt(d1) - std::sqrt(d2);
}

float Agent::reward1(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    float r = std::sqrt(d1) - std::sqrt(d2);
    return r/std::sqrt(1 + r*r);
}

float Agent::reward2(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    float r = 0;
    if (d1 > d2) {
        r = 0.8f;
    } else {
        r = -0.8f;
    }
    return r;
}

float Agent::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = std::sqrt((xi - xt) * (xi - xt) + (yi - yt) * (yi - yt));
    float d2 = std::sqrt((xn - xt) * (xn - xt) + (yn - yt) * (yn - yt));
    float r = (1 - 2*d2 + d2*d2)/(1 - d2 + d2*d2);
    if (d2 - d1 > 0) {
        r *= -1;
    }
    return r;
}

float Agent::reward4(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    return 0.01;
}

void Agent::observe(RL::Mat& statex, int x, int y, int xt, int yt)
{
    float xc = float(map.rows) / 2;
    float yc = float(map.cols) / 2;
    statex[0] = (x - xc) / xc;
    statex[1] = (y - yc) / yc;
    statex[2] = (xt - xc) / xc;
    statex[3] = (yt - yc) / yc;
    return;
}

int Agent::astarAction(int x, int y, int xt, int yt, float &totalReward)
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

int Agent::randAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    int direct = 0;
    float gamma = 0.9f;
    float T = 10000;
    RL::Mat a(4, 1);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            simulateMove(xn, yn, direct);
            a[direct] = gamma * a[direct] + reward0(xi, yi, xn, yn, xt, yt);
            if ((map(xn, yn) == 1) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal Action */
        direct = a.argmax();
        simulateMove(xn, yn, direct);
        if (map(xn, yn) != 1) {
            break;
        }
        a[direct] *= -2;
        T *= 0.98;
    }
    return direct;
}

int Agent::dqnAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Mat state0 = state;
    if (trainFlag == true) {
        float total = 0;
        for (int i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Mat& a = dqn.noiseAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            observe(nextState, xn, yn, xt, yt);
            total += r;
            if (map(xn, yn) == 1) {
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
        totalReward = total;
        /* training */
        dqn.learn(8192, 256, 64, 1e-3);
    }
    /* making decision */
    RL::Mat& a = dqn.action(state0);
    a.show();
    return a.argmax();
}

int Agent::qlstmAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        float total = 0;
        for (std::size_t i = 0; i < 256; i++) {
            int xi = xn;
            int yi = yn;
            RL::Mat& a = qlstm.eGreedyAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
                qlstm.perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                qlstm.perceive(state, a, nextState, r, true);
                break;
            } else {
                qlstm.perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        qlstm.learn(8192, 256, 16, 1e-3);
    }
    /* making decision */
    RL::Mat &a = qlstm.action(state_);
    a.show();
    return a.argmax();
}

int Agent::dpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        /* exploring environment */
        std::vector<RL::Step> steps;
        float total = 0;
        for (std::size_t i = 0; i < 16; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Mat &a = dpg.gumbelMax(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            steps.push_back(RL::Step(state, a, r));
            if (map(xn, yn) == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        dpg.reinforce(steps, 1e-3);
    }
    /* making decision */
    RL::Mat& a = dpg.action(state_);
    a.show();
    return a.argmax();
}

int Agent::drpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;  
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        std::vector<RL::Mat> states;
        std::vector<RL::Mat> actions;
        std::vector<float> rewards;
        float total = 0;
        for (std::size_t i = 0; i < 16; i++) {
            int xi = xn;
            int yi = yn;
            /* move */
            RL::Mat &a = drpg.noiseAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            /* sample */
            states.push_back(state);
            actions.push_back(a);
            rewards.push_back(r);
            total += r;
            if (map(xn, yn) == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        drpg.reinforce(states, actions, rewards, 1e-3);
    }
    /* making decision */
    RL::Mat &a = drpg.action(state_);
    a.show();
    return a.argmax();
}

int Agent::ddpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    float r = 0;
    float total = 0;
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        for (std::size_t i = 0; i < 16; i++) {
            int xi = xn;
            int yi = yn;
            RL::Mat & a = ddpg.noiseAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
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
        totalReward = total;
        /* training */
        ddpg.learn(8192, 256, 32, 0.001, 0.001);
    }
    return ddpg.action(state_);
}

int Agent::ppoAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        float total = 0;
        std::vector<RL::Step> trajectory;
        for (std::size_t i = 0; i < 32; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Mat &a = ppo.gumbelMax(state);
            int k = a.argmax();
            /* move */
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            trajectory.push_back(RL::Step(state, a, r));
            total += r;
            if (map(xn, yn) == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
#if 1
        ppo.learnWithClipObjective(trajectory, 1e-3);
#else
        ppo.learnWithKLpenalty(trajectory, 1e-3);
#endif
    }
    /* making decision */
    RL::Mat &a = ppo.action(state_);
    a.show();
    return a.argmax();
}

int Agent::sacAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Mat state_ = state;
    if (trainFlag == true) {
        float total = 0;
        for (int i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Mat& a = sac.gumbelMax(state);
            //int k = a.argmax();
            int k = RL::Random::categorical(a);
            simulateMove(xn, yn, k);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
                sac.perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                sac.perceive(state, a, nextState, r, true);
                break;
            } else {
                sac.perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        sac.learn(8192, 256, 64, 1e-3);
    }
    /* making decision */
    RL::Mat& a = sac.action(state_);
    a.show();
    return a.argmax();
}

int Agent::supervisedAction(int x, int y, int xt, int yt, float &totalReward)
{
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    float m = 0;
    RL::Mat& action = bpnn.output();
    if (trainFlag == true) {
        observe(state, xn, yn, xt, yt);
        for (std::size_t i = 0; i < 128; i++) {
            const RL::Mat &out = bpnn.forward(state);
            direct1 = action.argmax();
            direct2 = astarAction(xn, yn, xt, yt, totalReward);
            if (direct1 != direct2) {
                RL::Mat target(4, 1);
                target[direct2] = 1;
                bpnn.backward(RL::Loss::MSE(out, target));
                bpnn.gradient(state, target);
                m++;
            }
            if ((xn == xt) && (yn == yt)) {
                break;
            }
            if (map(xn, yn) == 1) {
                break;
            }
            observe(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            bpnn.optimize(RL::OPT_RMSPROP, 0.01);
        }
    }
    observe(state, x, y, xt, yt);
    bpnn.forward(state);
    direct1 = action.argmax();
    bpnn.show();
    return direct1;
}

bool Agent::simulateMove(int& x, int& y, int direct)
{
    moving(x, y, direct);
    bool flag = true;
    if (map(x, y) == 1) {
        flag = false;
    }
    return flag;
}
