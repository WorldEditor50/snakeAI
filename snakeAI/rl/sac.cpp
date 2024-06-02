#include "sac.h"

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.01*std::log(0.01);
    actor = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                 Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                 Softmax::_(hiddenDim, actionDim, true));

    int criticStateDim = stateDim;
    critic1Net = BPNN(Layer<Tanh>::_(criticStateDim, hiddenDim, true),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    critic1TargetNet = BPNN(Layer<Tanh>::_(criticStateDim, hiddenDim, false),
                            TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                            TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, false));

    critic2Net = BPNN(Layer<Tanh>::_(criticStateDim, hiddenDim, true),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    critic2TargetNet = BPNN(Layer<Tanh>::_(criticStateDim, hiddenDim, false),
                            TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                            TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, false));

    critic1Net.copyTo(critic1TargetNet);
    critic2Net.copyTo(critic2TargetNet);
}

void RL::SAC::perceive(const Mat& state,
                       const Mat& action,
                       const Mat& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void RL::SAC::perceive(const std::vector<RL::Transition> &x)
{
    for (int t = 0; t >= 0; t--) {
        memories.push_back(x[t]);
    }
    return;
}

RL::Mat &RL::SAC::eGreedyAction(const RL::Mat &state)
{
    Mat& out = actor.forward(state);
    return eGreedy(out, exploringRate, true);
}

RL::Mat &RL::SAC::gumbelMax(const RL::Mat &state)
{
    Mat& out = actor.forward(state);
    return RL::gumbelSoftmax(out, alpha.val);
}

RL::Mat& RL::SAC::action(const RL::Mat &state)
{
    return actor.forward(state);
}

void RL::SAC::experienceReplay(const RL::Transition &x)
{
    std::size_t i = x.action.argmax();
    /* train critic net */
    {
        /* select action */
        Mat& nextProb = actor.forward(x.nextState);
        std::size_t k = nextProb.argmax();
        /* select value */;
        Mat& q1 = critic1TargetNet.forward(x.nextState);
        Mat& q2 = critic2TargetNet.forward(x.nextState);
        float q = std::min(q1[k], q2[k]);
        float nextValue = 0;
        nextValue = nextProb[k]*(q - alpha[k]*std::log(nextProb[k] + 1e-8));
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*nextValue;
        }
        Mat qTarget1 = critic1Net.forward(x.state);
        qTarget1[i] = qTarget;
        critic1Net.backward(Loss::MSE(critic1Net.output(), qTarget1));
        critic1Net.gradient(x.state, qTarget1);
        Mat qTarget2 = critic2Net.forward(x.state);
        qTarget2[i] = qTarget;
        critic2Net.backward(Loss::MSE(critic2Net.output(), qTarget2));
        critic2Net.gradient(x.state, qTarget2);
    }
    /* train policy net */
    {
        Mat& q1 = critic1Net.forward(x.state);
        Mat& q2 = critic2Net.forward(x.state);
        float q = std::min(q1[i], q2[i]);
        const Mat& prob = actor.forward(x.state);
        Mat p(actionDim, 1);
        p[i] = prob[i]*(q - alpha[i]*std::log(prob[i] + 1e-8));
        actor.backward(Loss::CrossEntropy(prob, p));
        actor.gradient(x.state, p);
    }
    /* alpha */
    {
        const Mat& prob = x.action;
        alpha.g[i] += -prob[i]*std::log(prob[i] + 1e-8) - entropy0;
    }
    return;
}

void RL::SAC::learn(size_t maxMemorySize, size_t replaceTargetIter, size_t batchSize, float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update */
        critic1Net.softUpdateTo(critic1TargetNet, 1e-3);
        critic2Net.softUpdateTo(critic2TargetNet, 2e-3);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }
    //actor.optimize(OPT_NORMRMSPROP, 1e-2);
    actor.optimize(OPT_NORMRMSPROP, 1e-3, 0.1);
    actor.clamp(-1, 1);
    alpha.RMSProp(0.9, 1e-5, 0);
    //alpha.clamp(0.1, 1.2);
#if 1
    std::cout<<"alpha:";
    alpha.val.show();
#endif
    critic1Net.optimize(OPT_NORMRMSPROP, 1e-3, 0);
    critic2Net.optimize(OPT_NORMRMSPROP, 1e-3, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.2 ? 0.2 : exploringRate;
    learningSteps++;
    return;
}

void RL::SAC::save()
{
    actor.save("sac_actor");
    critic1Net.save("sca_critic1");
    critic2Net.save("sca_critic2");
    return;
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    critic1Net.load("sca_critic1");
    critic2Net.load("sca_critic2");
    critic1Net.copyTo(critic1TargetNet);
    critic2Net.copyTo(critic2TargetNet);
    return;
}
