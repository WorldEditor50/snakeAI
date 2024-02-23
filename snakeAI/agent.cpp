#include "agent.h"
#include <QDebug>
Agent::Agent(Mat& map, Snake &s):
    map(map),
    snake(s),
    trainFlag(true)
{
    int stateDim = 4;
    dqn = DQN(stateDim, 16, 4);
    dpg = DPG(stateDim, 16, 4);
    ddpg = DDPG(stateDim, 16, 4);
    ppo = PPO(stateDim, 16, 4);
    sac = SAC(stateDim, 16, 4);
    bpnn = BPNN(Layer<Sigmoid>::_(stateDim, 16, true),
                Layer<Sigmoid>::_(16, 16, true),
                Layer<Sigmoid>::_(16, 16, true),
                Layer<Sigmoid>::_(16, 4, true));
    qlstm = QLSTM(stateDim, 16, 4);
    drpg = DRPG(stateDim, 16, 4);
    state = Mat(stateDim, 1);
    nextState = Mat(stateDim, 1);
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

float Agent::reward0(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
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
        r = -1;
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
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return std::tanh(1/(d1 - d2));
}

void Agent::observe(Mat& statex, int x, int y, int xt, int yt)
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
    Mat act(4, 1);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            simulateMove(xn, yn, direct);
            act[direct] = gamma * act[direct] + reward0(xi, yi, xn, yn, xt, yt);
            if ((map(xn, yn) == 1) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal Action */
        direct = act.argmax();
        simulateMove(xn, yn, direct);
        if (map(xn, yn) != 1) {
            break;
        }
        /* punishment */
        act[direct] *= -2;
        T = 0.98 * T;
    }
    return direct;
}

int Agent::dqnAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    Mat state0 = state;
    if (trainFlag == true) {
        int i = 0;
        float total = 0;
        for (i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            Mat& action = dqn.noiseAction(state);
            int k = action.argmax();
            simulateMove(xn, yn, k);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            observe(nextState, xn, yn, xt, yt);
            total += r;
            if (map(xn, yn) == 1) {
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
        totalReward = total;
        /* training */
        dqn.learn(OPT_RMSPROP, 8192, 256, 64, 1e-3);
    }
    /* making decision */
    return dqn.action(state0);
}

int Agent::qlstmAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    float r = 0;
    float total = 0;
    observe(state, x, y, xt, yt);
    Mat state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 256; j++) {
            int xi = xn;
            int yi = yn;
            Mat& action = qlstm.eGreedyAction(state);
            int k = action.argmax();
            simulateMove(xn, yn, k);
            r = reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
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
        totalReward = total;
        /* training */
        qlstm.learn(8192, 256, 16, 1e-3);
    }
    /* making decision */
    Mat &action = qlstm.action(state_);
    for (std::size_t i = 0; i < action.size(); i++) {
        std::cout<<action[i]<<" ";
    }
    std::cout<<std::endl;
    return action.argmax();
}

int Agent::dpgAction(int x, int y, int xt, int yt, float &totalReward)
{

    int direct = 0;
    /* exploring environment */
    std::vector<Step> steps;
    int xn = x;
    int yn = y;
    float total = 0;
    observe(state, x, y, xt, yt);
    Mat state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 16; j++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            Mat &output = dpg.gumbelMax(state);
            direct = output.argmax();
            simulateMove(xn, yn, direct);
            observe(nextState, xn, yn, xt, yt);
            Step s;
            s.state = state;
            s.action  = output;
            s.reward  = reward0(xi, yi, xn, yn, xt, yt);
            total += s.reward;
            steps.push_back(s);
            if (map(xn, yn) == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        dpg.reinforce(OPT_RMSPROP, 1e-5, steps);
    }
    /* making decision */
    direct = dpg.action(state_);
    return direct;
}

int Agent::drpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int direct = 0;
    std::vector<Mat> states;
    std::vector<Mat> actions;
    std::vector<float> rewards;
    int xn = x;
    int yn = y;
    float total = 0;
    observe(state, x, y, xt, yt);
    Mat state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 16; j++) {
            int xi = xn;
            int yi = yn;
            /* move */
            Mat &output = drpg.gumbelMax(state);
            direct = output.argmax();
            simulateMove(xn, yn, direct);
            observe(nextState, xn, yn, xt, yt);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            /* sample */
            states.push_back(state);
            actions.push_back(output);
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
    Mat &a = drpg.action(state_);
    for (std::size_t i = 0; i < a.size(); i++) {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
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
    Mat state_ = state;
    if (trainFlag == true) {
        for (std::size_t j = 0; j < 16; j++) {
            int xi = xn;
            int yi = yn;
            Mat & action = ddpg.noiseAction(state);
            int k = action.argmax();
            simulateMove(xn, yn, k);
            r = reward1(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
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
        totalReward = total;
        /* training */
        ddpg.learn(OPT_RMSPROP, 8192, 256, 32, 0.001, 0.001);
    }
    return ddpg.action(state_);
}

int Agent::ppoAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    float total = 0;
    observe(state, x, y, xt, yt);
    Mat state_ = state;
    if (trainFlag == true) {
        std::vector<Step> trajectories;
        for (std::size_t j = 0; j < 32; j++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            Mat &output = ppo.gumbelMax(state);
            int direct = output.argmax();
            /* move */
            simulateMove(xn, yn, direct);
            observe(nextState, xn, yn, xt, yt);
            float r = reward0(xi, yi, xn, yn, xt, yt);
            trajectories.push_back(Step(state, output, r));
            total += r;
            if (map(xn, yn) == 1 || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
#if 1
        ppo.learnWithClipObjective(1e-3, trajectories);
#else
        ppo.learnWithKLpenalty(1e-3, trajectories);
#endif
    }
    /* making decision */
    Mat &a = ppo.action(state_);
    for (std::size_t i = 0; i < a.size(); i++) {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
    return a.argmax();
}

int Agent::sacAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    float r = 0;
    float total = 0;
    observe(state, x, y, xt, yt);
    Mat state_ = state;
    int i = 0;
    if (trainFlag == true) {
        for (i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            Mat& action = sac.eGreedyAction(state);
            int k = action.argmax();
            simulateMove(xn, yn, k);
            r = reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (map(xn, yn) == 1) {
                sac.perceive(state, action, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                sac.perceive(state, action, nextState, r, true);
                break;
            } else {
                sac.perceive(state, action, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        sac.learn(OPT_RMSPROP, 8192, 256, 64, 1e-3);
    }
    /* making decision */
    Mat& a = sac.action(state_);
#if 0
    for (std::size_t i = 0; i < a.size(); i++) {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
#endif
    return a.argmax();
}

int Agent::supervisedAction(int x, int y, int xt, int yt, float &totalReward)
{
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    float m = 0;
    Mat& action = bpnn.output();
    if (trainFlag == true) {
        observe(state, xn, yn, xt, yt);
        for (std::size_t i = 0; i < 128; i++) {
            bpnn.forward(state);
            direct1 = action.argmax();
            direct2 = astarAction(xn, yn, xt, yt, totalReward);
            if (direct1 != direct2) {
                Mat target(4, 1);
                target[direct2] = 1;
                bpnn.gradient(state, target, Loss::MSE);
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
            bpnn.optimize(OPT_RMSPROP, 0.01);
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
