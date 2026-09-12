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
    trpo = RL::TRPO(stateDim, 16, 4);
    sac = RL::SAC(stateDim, 16, 4);
    bpnn = RL::Net(RL::Layer<RL::Sigmoid>::_(stateDim, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 16, true, true),
                   RL::Layer<RL::Sigmoid>::_(16, 4, true, true));
    drpg = RL::DRPG(stateDim, 16, 4);
    convpg = RL::ConvPG(stateDim, 16, 4);
    convdqn = RL::ConvDQN(stateDim, 64, 4);
    bcq = RL::BCQ(stateDim, 16, 4);
    mpg = RL::MPG(stateDim, 16, 4);

    state = RL::Tensor(stateDim, 1);
    nextState = RL::Tensor(stateDim, 1);
    //dqn.load("./dqn");
    //dpg.load("./dpg");
    //ddpg.load("./ddpg_actor", "./ddpg_critic");
    //bpnn.load("./bpnn");
    //ppo.load("./ppo_actor", "./ppo_critic");
    //sac.load();
}

Agent::~Agent()
{
    dqn.save("./dqn");
    dpg.save("./dpg");
    ddpg.save("./ddpg_actor", "./ddpg_critic");
    //bpnn.save("./bpnn");
    ppo.save("./ppo_actor", "./ppo_critic");
    trpo.save("./trpo_actor", "./trpo_critic");
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
        dqn.learn(16384, 256, 64, 1e-3);
    }
    /* making decision */
    RL::Tensor& a = dqn.action(state0);
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
        /*
           The game trains with reinforce(), not reinforce1().

           Measured in the running game, reinforce() produces visibly better play
           than reinforce1(). The two are NOT the same estimator. With
           CrossEntropy::df(out, target)[i] = -target[i]/out[i] and reinforce()'s
           in-place edit x[t].action[k] = p_k * A_t, the resulting logit update is

               reinforce()  :  dz = eta * p_k * A_t * (e_k - pi)
               reinforce1() :  dz = eta *        A_t * (e_k - pi)

           where p_k is the probability the sampled Gumbel-Softmax distribution
           gave to the action that was actually taken. reinforce() therefore
           weights every update by how confident the policy was about that
           action, which damps steps taken when the choice was close to a coin
           flip.

           reinforce1() remains the exact, side-effect-free estimator and is the
           one the test suite pins down (test_pg / test_mpg / test_convpg), so
           both implementations stay in the library. */
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
        /* Each rollout is an independent episode, so the LSTM state must start
           at zero — both to stop one episode's context leaking into the next and
           because reinforce()/reinforce1() replay this trajectory from a RESET
           state: if the rollout began from a dirty state, the forward passes used
           to build the gradient would not correspond to the policy that produced
           the actions. */
        drpg.resetState();
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
        /* reinforce(), as measured to play better than reinforce1(); the
           difference between the two is derived in dpgAction() above.
           rl/drpg.* keeps both, and the test suite covers reinforce1(). */
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
        /* reinforce(), as measured to play better than reinforce1(); see the
           derivation in dpgAction(). rl/convpg.* keeps both. */
        convpg.reinforce(steps, 1e-2);
    }
    /* making decision */
    RL::Tensor& a = convpg.action(state_);
    return a.argmax();
}

int Agent::convdqnAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    RL::Tensor cloneMap = env.map;
    Snake cloneSnake(snake.body, cloneMap);
    /*
       Normalise by a CONSTANT, not by the board's own maximum. The map holds
       OBJ_NONE=0, OBJ_BLOCK=1, OBJ_SNAKE=2, OBJ_TARGET=4, so dividing by
       state.max() happened to give 4 whenever the target was present — but the
       encoding of every cell would silently change the moment it was not. A
       fixed divisor keeps the input representation stationary.
    */
    state = cloneMap;
    state /= 4.0f;
    state.reshape(1, 118, 118);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        /* exploring environment */
        float total = 0;
        for (std::size_t i = 0; i < 64; i++) {
            int xi = xn;
            int yi = yn;
            /*
               epsilon-greedy, not noiseAction(). noise() adds U(0,2) to every
               Q-value and renormalises by the maximum, which randomises the
               argmax outright; it was also driven by an epsilon that never
               decayed (see ConvDQN::learn), so the agent explored forever.
            */
            RL::Tensor &a = convdqn.eGreedyAction(state);
            int k = a.argmax();
            simulateMove(cloneSnake, xn, yn, k);
            nextState = cloneMap;
            nextState /= 4.0f;
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
    return a.argmax();
}

int Agent::trpoAction(int x, int y, int xt, int yt, float &totalReward)
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
            RL::Tensor &a = trpo.eGreedyAction(state);
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
        trpo.learn(trajectory, 1e-3);
    }
    /* making decision */
    RL::Tensor &a = trpo.action(state_);
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
#if 0
            const RL::Tensor& prob = sac.action(state);
            int k = RL::Random::categorical(prob);
#else
            RL::Tensor &prob = sac.gumbelMax(state);
            int k = prob.argmax();
#endif
            simulateMove(xn, yn, k);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (env.map(xn, yn) == OBJ_BLOCK) {
                sac.perceive(state, prob, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                sac.perceive(state, prob, nextState, r, true);
                break;
            } else {
                sac.perceive(state, prob, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        sac.learn(16384, 256, 64, 1e-3);
    }
    /* making decision */
    RL::Tensor& a = sac.action(state_);
    //a.printValue();
    return a.argmax();
}

int Agent::bcqAction(int x, int y, int xt, int yt, float &totalReward)
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
            /* BCQ: VAE generates candidate actions, actor refines and adds noise */
            const RL::Tensor& prob = bcq.action(state);
            int k = RL::Random::categorical(prob);
            simulateMove(xn, yn, k);
            float r = env.reward0(xi, yi, xn, yn, xt, yt);
            total += r;
            observe(nextState, xn, yn, xt, yt);
            if (env.map(xn, yn) == OBJ_BLOCK) {
                bcq.perceive(state, prob, nextState, r, true);
                break;
            }
            if (xn == xt && yn == yt) {
                bcq.perceive(state, prob, nextState, r, true);
                break;
            } else {
                bcq.perceive(state, prob, nextState, r, false);
            }
            state = nextState;
        }
        totalReward = total;
        /* training */
        bcq.learn(8192, 256, 32, 1e-3);
    }
    /* making decision */
    RL::Tensor& a = bcq.action(state_);
    //a.printValue();
    return a.argmax();
}

int Agent::mpgAction(int x, int y, int xt, int yt, float &totalReward)
{
    int xn = x;
    int yn = y;
    observe(state, x, y, xt, yt);
    RL::Tensor state_ = state;
    if (trainFlag == true) {
        std::vector<RL::Step> steps;
        float total = 0;
        /* Fresh episode => fresh recurrent state. See the note in drpgAction():
           reinforce()/reinforce1() replay this trajectory from a reset Mamba
           state, so the rollout has to start from one as well. */
        mpg.resetState();
        for (std::size_t i = 0; i < 32; i++) {
            int xi = xn;
            int yi = yn;
            /* move using Gumbel-Softmax exploration */
            RL::Tensor &a = mpg.gumbelMax(state);
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
        /* reinforce(), as measured to play better than reinforce1(); see the
           derivation in dpgAction(). rl/mpg.* keeps both. */
        mpg.reinforce(steps, 1e-2);
    }
    /* making decision */
    RL::Tensor &a = mpg.action(state_);
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
                bpnn.backward(state, RL::Loss::MSE::df(out, target));
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
            /* Net::RMSProp(lr, rho, decay): the arguments were swapped, so the
               BPNN was trained with lr=0.9 (divergent) and rho=1e-3. */
            bpnn.RMSProp(1e-3, 0.9, 0.1);
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
