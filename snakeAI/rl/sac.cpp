#include "sac.h"

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(-4);
    actor = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                 LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                 Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                 LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                 SoftmaxLayer::_(hiddenDim, actionDim, true));

    critic1Net = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    critic1TargetNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, false));

    critic2Net = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    critic2TargetNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, false));

    critic1Net.copyTo(critic1TargetNet);
    critic2Net.copyTo(critic2TargetNet);
}

void RL::SAC::perceive(const RL::Mat &state,
                       const RL::Mat &action,
                       const RL::Mat &nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Mat &RL::SAC::eGreedyAction(const RL::Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        actor.output().zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        actor.output()[index] = 1;
    } else {
        actor.forward(state);
    }
    return actor.output();
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
        /* select value */
        Mat& q1 = critic1TargetNet.forward(x.nextState);
        Mat& q2 = critic2TargetNet.forward(x.nextState);
        float q = std::min(q1[k], q2[k]);
        float nextValue = -std::exp(alpha[k])*nextProb[k]*std::log(nextProb[k]) + nextProb[k]*q;
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*nextValue;
        }
        Mat qTarget1 = critic1Net.forward(x.state);
        qTarget1[i] = qTarget;
        critic1Net.gradient(x.state, qTarget1, Loss::MSE);
        Mat qTarget2 = critic2Net.forward(x.state);
        qTarget2[i] = qTarget;
        critic2Net.gradient(x.state, qTarget2, Loss::MSE);
    }
    /* train policy net */
    {
        const Mat& prob = x.action;
        Mat& q1 = critic1Net.forward(x.state);
        Mat& q2 = critic2Net.forward(x.state);
        float q = std::min(q1[i], q2[i]);
        Mat target = prob;
        target[i] *= -std::exp(alpha[i])*prob[i]*std::log(prob[i]) + prob[i]*q;
        actor.gradient(x.state, target, Loss::CrossEntropy);
        //target[i] = std::exp(q) - prob[i];
        //actor.gradient(x.state, target, Loss::MSE);
        /* alpha -> 0 */
        alpha.d[i] = std::exp(alpha[i])*(-prob[i]*std::log(prob[i]) + 1)*alpha[i];
    }
    return;
}

void RL::SAC::learn(RL::OptType optType, size_t maxMemorySize, size_t replaceTargetIter, size_t batchSize, float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update */
        critic1Net.softUpdateTo(critic1TargetNet, 0.01);
        critic2Net.softUpdateTo(critic2TargetNet, 0.001);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Rand::engine);
        experienceReplay(memories[k]);
    }
    actor.optimize(optType, 1e-3, 0.1);
    actor.clamp(-1, 1);
    alpha.RMSProp(0.9, 1e-2, 0.1);
#if 0
    for (std::size_t i = 0; i < alpha.val.size(); i++) {
        std::cout<<alpha.val[i]<<" ";
    }
    std::cout<<std::endl;
#endif
    critic1Net.optimize(optType, 1e-3, 0);
    critic2Net.optimize(optType, 1e-3, 0);

    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
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
