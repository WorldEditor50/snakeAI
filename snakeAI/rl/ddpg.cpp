#include "ddpg.h"

RL::DDPG::DDPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.99;
    beta = 1;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    /* actor: a = P(s, theta) */
    actorP = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                  LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                  Softmax::_(hiddenDim, actionDim, true));

    actorQ = BPNN(Layer<Tanh>::_(stateDim, hiddenDim,false),
                  LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, false),
                  Layer<Tanh>::_(hiddenDim, hiddenDim,false),
                  LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, false),
                  Softmax::_(hiddenDim, actionDim, false));
    actorP.copyTo(actorQ);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    criticP = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    criticQ = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, false),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, false));
    criticP.copyTo(criticQ);
}

void RL::DDPG::perceive(const Mat& state,
                        const Mat &action,
                        const Mat& nextState,
                        float reward,
                        bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Mat& RL::DDPG::noiseAction(const Mat &state)
{
    Mat& out = actorP.forward(state);
    return noise(out);
}

RL::Mat& RL::DDPG::action(const Mat &state)
{
    return actorP.forward(state);
}

void RL::DDPG::experienceReplay(const Transition& x)
{
    /* train critic */
    std::size_t i = x.action.argmax();
    {
        Mat &ap = actorP.forward(x.state);
        Mat sa(stateDim + actionDim, 1);
        Mat::concat(0, sa, x.state, ap);
        Mat ct = criticP.forward(sa);
        if (x.done == true) {
            ct[i] = x.reward;
        } else {
            Mat &aq = actorQ.forward(x.nextState);
            int k = aq.argmax();
            Mat::concat(0, sa, x.nextState, aq);
            Mat &cq = criticQ.forward(sa);
            ct[k] = x.reward + gamma*cq[k];
        }
        Mat::concat(0, sa, x.state, ap);
        Mat &cp = criticP.forward(sa);
        criticP.backward(Loss::MSE(cp, ct));
        criticP.gradient(sa, ct);
    }

    /* train actor */
    {
        Mat &ap = actorP.forward(x.state);
        Mat sa(stateDim + actionDim, 1);
        Mat::concat(0, sa, x.state, ap);
        Mat& q = criticP.forward(sa);
        Mat at(actionDim, 1);
        at[i] = ap[i]*q[i];
        actorP.backward(Loss::CrossEntropy(ap, at));
        actorP.gradient(x.state, at);
    }

    return;
}

void RL::DDPG::learn(std::size_t maxMemorySize,
                     std::size_t replaceTargetIter,
                     std::size_t batchSize)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update critic */
        criticP.softUpdateTo(criticQ, 0.01);
        learningSteps = 0;
        std::cout<<"update target net"<<std::endl;
        /* update actor */
        actorP.softUpdateTo(actorQ, 0.01);
    }
    /* experience replay */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Random::engine);
        experienceReplay(memories[k]);
    }
    actorP.optimize(OPT_NORMRMSPROP, 1e-3, 0.1);
    actorP.clamp(-1, 1);
    criticP.optimize(OPT_NORMRMSPROP, 1e-3, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::DDPG::save(const std::string& actorPara, const std::string& criticPara)
{
    actorP.save(actorPara);
    criticP.save(criticPara);
    return;
}

void RL::DDPG::load(const std::string& actorPara, const std::string& criticPara)
{
    actorP.load(actorPara);
    actorP.copyTo(actorQ);
    criticP.load(criticPara);
    criticP.copyTo(criticQ);
    return;
}
