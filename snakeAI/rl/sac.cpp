#include "sac.h"
#include "layer.h"
#include "loss.h"

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.08*std::log(0.08);
    actor = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    int criticStateDim = stateDim;
    critic1Net = Net(Layer<Tanh>::_(criticStateDim, hiddenDim, true, true),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    critic1TargetNet = Net(Layer<Tanh>::_(criticStateDim, hiddenDim, true, false),
                           TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                           Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                           TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                           Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));

    critic2Net = Net(Layer<Tanh>::_(criticStateDim, hiddenDim, true, true),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    critic2TargetNet = Net(Layer<Tanh>::_(criticStateDim, hiddenDim, true, false),
                           TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                           Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                           TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                           Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));

    critic1Net.copyTo(critic1TargetNet);
    critic2Net.copyTo(critic2TargetNet);
}

void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
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

RL::Tensor &RL::SAC::eGreedyAction(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return eGreedy(out, exploringRate, true);
}

RL::Tensor &RL::SAC::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return RL::gumbelSoftmax(out, alpha.val);
}

RL::Tensor& RL::SAC::action(const RL::Tensor &state)
{
    return actor.forward(state);
}

void RL::SAC::experienceReplay(const RL::Transition &x)
{
    std::size_t i = x.action.argmax();
    /* train critic net */
    {
        /* select action */
        Tensor& nextProb = actor.forward(x.nextState);
        std::size_t k = nextProb.argmax();
        /* select value */;
        Tensor& q1 = critic1TargetNet.forward(x.nextState);
        Tensor& q2 = critic2TargetNet.forward(x.nextState);
        float q = std::min(q1[k], q2[k]);
        float nextValue = 0;
        nextValue = nextProb[k]*(q - alpha[k]*std::log(nextProb[k] + 1e-8));
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*nextValue;
        }
        Tensor out1 = critic1Net.forward(x.state);
        Tensor qTarget1 = out1;
        qTarget1[i] = qTarget;
        critic1Net.backward(Loss::MSE(out1, qTarget1));
        critic1Net.gradient(x.state, qTarget1);
        Tensor out2 = critic2Net.forward(x.state);
        Tensor qTarget2 = out2;
        qTarget2[i] = qTarget;
        critic2Net.backward(Loss::MSE(out2, qTarget2));
        critic2Net.gradient(x.state, qTarget2);
    }
    /* train policy net */
    {
        const Tensor &sampleProb = x.action;
        const Tensor& prob = actor.forward(x.state);
        //const Tensor &prob = x.action;
        //int i = prob.argmax();
        Tensor& q1 = critic1Net.forward(x.state);
        Tensor& q2 = critic2Net.forward(x.state);
        float q = std::min(q1[i], q2[i]);

        Tensor loss(actionDim, 1);
#if 0
        /* importance sampling */
        float r = prob[i]/(sampleProb[i] + 1e-8);
        loss[i] = r*(0.9*q - alpha[i]*std::log(r) + std::log(sampleProb[i]));
#else
        loss[i] = prob[i]*(q - alpha[i]*std::log(prob[i]));
#endif
        actor.backward(loss);
        actor.gradient(x.state, loss);
    }
    /* alpha */
    {
        const Tensor& prob = x.action;
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
    actor.RMSProp(1e-3, 0.9, 0.1);
    alpha.RMSProp(1e-4, 0.9, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    critic1Net.RMSProp(1e-3, 0.9, 0);
    critic2Net.RMSProp(1e-3, 0.9, 0);
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
    //actor.save("sac_actor");
    //critic1Net.save("sca_critic1");
    //critic2Net.save("sca_critic2");
    return;
}

void RL::SAC::load()
{
    //actor.load("sac_actor");
    //critic1Net.load("sca_critic1");
    //critic2Net.load("sca_critic2");
    //critic1Net.copyTo(critic1TargetNet);
    //critic2Net.copyTo(critic2TargetNet);
    return;
}
