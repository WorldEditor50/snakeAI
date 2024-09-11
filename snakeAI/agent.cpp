#include "agent.h"
#include "environment.h"
#include "rl/layer.h"
Agent::Agent(Environment& env_, Snake &s):
    env(env_), snake(s),
    trainFlag(true)
{
    int stateDim = 4;
    dqn = RL::DQN(stateDim, 16, 4);
    dpg = RL::DPG(stateDim, 16, 4);
    ddpg = RL::DDPG(stateDim, 16, 4);
    ppo = RL::PPO(stateDim, 16, 4);
    sac = RL::SAC(stateDim, 16, 4);
    bpnn = RL::Net(RL::Layer<RL::Sigmoid>::_(stateDim, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 4, true, true));
    qlstm = RL::QLSTM(stateDim, 16, 4);
    drpg = RL::DRPG(stateDim, 16, 4);
    convpg = RL::ConvPG(stateDim, 16, 4);
    convdqn = RL::ConvDQN(stateDim, 16, 4);
    state = RL::Tensor(stateDim, 1);
    nextState = RL::Tensor(stateDim, 1);
    dqn.load("./dqn");
    dpg.load("./dpg");
    ddpg.load("./ddpg_actor", "./ddpg_critic");
    bpnn.load("./bpnn");
    ppo.load("./ppo_actor", "./ppo_critic");
    sac.load();
}

Agent::~Agent()
{
    dqn.save("./dqn");
    dpg.save("./dpg");
    ddpg.save("./ddpg_actor", "./ddpg_critic");
    //bpnn.save("./bpnn");
    ppo.save("./ppo_actor", "./ppo_critic");
    sac.save();
}


void Agent::observe(RL::Tensor& statex, int x, int y, int xt, int yt)
{
    float xc = float(env.map.shape[0]) / 2;
    float yc = float(env.map.shape[1]) / 2;
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
    RL::Tensor a(4, 1);
    while (T > 0.001) {
        /* do experiment */
        while (T > 0.01) {
            direct = rand() % 4;
            int xi = xn;
            int yi = yn;
            simulateMove(xn, yn, direct);
            a[direct] = gamma * a[direct] + env.reward0(xi, yi, xn, yn, xt, yt);
            if ((env.map(xn, yn) == OBJ_BLOCK) || (xn == xt && yn == yt)) {
                break;
            }
        }
        xn = x;
        yn = y;
        /* select optimal Action */
        direct = a.argmax();
        simulateMove(xn, yn, direct);
        if (env.map(xn, yn) != OBJ_BLOCK) {
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
    RL::Tensor state0 = state;
    if (trainFlag == true) {
        float total = 0;
        for (int i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Tensor& a = dqn.noiseAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            observe(nextState, xn, yn, xt, yt);
            total += r;
            if (env.map(xn, yn) == OBJ_BLOCK) {
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
    RL::Tensor& a = dqn.action(state0);
    a.printValue();
    return a.argmax();
}

int Agent::qlstmAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        float total = 0;
        for (std::size_t i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Tensor& a = qlstm.noiseAction(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (env.map(xn, yn) == OBJ_BLOCK) {
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
    RL::Tensor &a = qlstm.action(state_);
    a.printValue();
    return a.argmax();
}

int Agent::dpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        /* exploring environment */
        std::vector<RL::Step> steps;
        float total = 0;
        for (std::size_t i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Tensor &a = dpg.gumbelMax(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            steps.push_back(RL::Step(state, a, r));
            if (env.map(xn, yn) == OBJ_BLOCK || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        dpg.reinforce(steps, 1e-2);
    }
    /* making decision */
    RL::Tensor& a = dpg.action(state_);
    //a.printValue();
    return a.argmax();
}

int Agent::drpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        std::vector<RL::Step> steps;
        float total = 0;
        for (std::size_t i = 0; i < 16; i++) {
            int xi = xn;
            int yi = yn;
            /* move */
            RL::Tensor &a = drpg.gumbelMax(state);
            int k = a.argmax();
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            /* sample */
            steps.push_back(RL::Step(state, a, r));
            total += r;
            if (env.map(xn, yn) == OBJ_BLOCK || (xn == xt && yn == yt)) {
                break;
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        drpg.reinforce(steps, 1e-2);
    }
    /* making decision */
    RL::Tensor &a = drpg.action(state_);
    //a.printValue();
    return a.argmax();
}

int Agent::convpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    RL::Tensor cloneMap = env.map;
    Snake cloneSnake(snake.body, cloneMap);
    state = cloneMap;
    state /= state.max();
    state.reshape(1, 118, 118);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        /* exploring environment */
        std::vector<RL::Step> steps;
        float total = 0;
        for (std::size_t i = 0; i < 16; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Tensor &a = convpg.gumbelMax(state);
            int k = a.argmax();
            simulateMove(cloneSnake, xn, yn, k);
            //float r = env.reward2(cloneMap, xi, yi, xn, yn, xt, yt);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            steps.push_back(RL::Step(state, a, r));
            if (cloneMap(xn, yn) == OBJ_BLOCK || (xn == xt && yn == yt)) {
                break;
            }
            state = cloneMap;
            state /= state.max();
            state.reshape(1, 118, 118);
        }
        totalReward = total;
        /* training */
        convpg.reinforce(steps, 1e-2);
    }
    /* making decision */
    RL::Tensor& a = convpg.action(state_);
    a.printValue();
    return a.argmax();
}

int Agent::convdqnAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    RL::Tensor cloneMap = env.map;
    Snake cloneSnake(snake.body, cloneMap);
    state = cloneMap;
    state /= state.max();
    state.reshape(1, 118, 118);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        /* exploring environment */
        float total = 0;
        for (std::size_t i = 0; i < 64; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Tensor &a = convdqn.noiseAction(state);
            int k = a.argmax();
            simulateMove(cloneSnake, xn, yn, k);
            nextState = cloneMap;
            nextState /= nextState.max();
            nextState.reshape(1, 118, 118);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            if (cloneMap(xn, yn) == OBJ_BLOCK) {
                convdqn.perceive(state, a, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                convdqn.perceive(state, a, nextState, r, true);
                break;
            } else {
                convdqn.perceive(state, a, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        convdqn.learn(4096, 256, 32, 1e-2);
    }
    /* making decision */
    RL::Tensor& a = convdqn.action(state_);
    a.printValue();
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
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        for (std::size_t i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Tensor & a = ddpg.gumbelMax(state);
            int k = RL::Random::categorical(a);
            simulateMove(xn, yn, k);
            r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (env.map(xn, yn) == OBJ_BLOCK) {
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
        ddpg.learn(8192, 256, 32);
    }

    RL::Tensor &a = ddpg.action(state_);
    a.printValue();
    return a.argmax();
}

int Agent::ppoAction(int x, int y, int xt, int yt, float &totalReward)
{
    /* exploring environment */
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        float total = 0;
        std::vector<RL::Step> trajectory;
        for (std::size_t i = 0; i < 32; i++) {
            int xi = xn;
            int yi = yn;
            /* sample */
            RL::Tensor &a = ppo.gumbelMax(state);
            int k = a.argmax();
            /* move */
            simulateMove(xn, yn, k);
            observe(nextState, xn, yn, xt, yt);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            trajectory.push_back(RL::Step(state, a, r));
            total += r;
            if (env.map(xn, yn) == OBJ_BLOCK || (xn == xt && yn == yt)) {
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
    RL::Tensor &a = ppo.action(state_);
    a.printValue();
    return a.argmax();
}

int Agent::sacAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        float total = 0;
        for (int i = 0; i < 128; i++) {
            int xi = xn;
            int yi = yn;
            RL::Tensor& a = sac.gumbelMax(state);
            //int k = a.argmax();
            int k = RL::Random::categorical(a);
            simulateMove(xn, yn, k);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            //float r = env.reward2(env.map, xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (env.map(xn, yn) == OBJ_BLOCK) {
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
        sac.learn(4096, 256, 64, 1e-3);
    }
    /* making decision */
    RL::Tensor& a = sac.action(state_);
    //a.printValue();
    return a.argmax();
}

int Agent::supervisedAction(int x, int y, int xt, int yt, float &totalReward)
{
    int direct1 = 0;
    int direct2 = 0;
    int xn = x;
    int yn = y;
    float m = 0;
    if (trainFlag == true) {
        observe(state, xn, yn, xt, yt);
        for (std::size_t i = 0; i < 128; i++) {
            const RL::Tensor &out = bpnn.forward(state);
            direct1 = out.argmax();
            direct2 = astarAction(xn, yn, xt, yt, totalReward);
            if (direct1 != direct2) {
                RL::Tensor target(4, 1);
                target[direct2] = 1;
                bpnn.backward(RL::Loss::MSE(out, target));
                bpnn.gradient(state, target);
                m++;
            }
            if ((xn == xt) && (yn == yt)) {
                break;
            }
            if (env.map(xn, yn) == OBJ_BLOCK) {
                break;
            }
            observe(state, xn, yn, xt, yt);
        }
        if (m > 0) {
            bpnn.RMSProp(0.9, 1e-3, 0.1);
        }
    }
    observe(state, x, y, xt, yt);
    direct1 = bpnn.forward(state).argmax();
    return direct1;
}

bool Agent::simulateMove(int& x, int& y, int direct)
{
    moving(x, y, direct);
    bool flag = true;
    if (env.map(x, y) == 1) {
        flag = false;
    }
    return flag;
}

void Agent::simulateMove(Snake &clone, int& x, int& y, int k)
{
    moving(x, y, k);
    clone.move(k);
    return;
}
