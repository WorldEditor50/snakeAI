#include "sac.h"
#include "layer.h"
#include "loss.h"

#define USE_DECAY 1

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    annealing = ExpAnnealing(0.01, 0.12, 1e-4);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 =  RL::entropy(0.1);
    actor = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    critic1 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                  Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));
    critic2 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                  Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    critic1Target = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                        TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                        Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
    critic2Target = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                        TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                        Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));

    critic1.copyTo(critic1Target);
    critic2.copyTo(critic2Target);
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

void RL::SAC::experienceReplay(const RL::Transition &x, float beta)
{
    /* train critic net */
#if 0
    {
        /* select action */
        const Tensor& nextProb = actor.forward(x.nextState);
        /* select value */;
        const Tensor& q1 = critic1Target.forward(x.nextState);
        const Tensor& q2 = critic2Target.forward(x.nextState);
        Tensor nextValue(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            nextValue[i] = nextProb[i]*(std::min(q1[i], q2[i]) - alpha[i]*std::log(nextProb[i] + 1e-8));
        }
        int k = x.action.argmax();
        Tensor reward(actionDim, 1);
        reward[k] = x.reward;
        Tensor qTarget(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            if (x.done) {
                qTarget[i] = reward[i];
            } else {
                qTarget[i] = reward[i] + gamma*nextValue[i];
            }
        }
        const Tensor &out1 = critic1.forward(x.state);
        critic1.backward(Loss::MSE::df(out1, qTarget));
        critic1.gradient(x.state, qTarget);
        const Tensor &out2 = critic2.forward(x.state);
        critic2.backward(Loss::MSE::df(out2, qTarget));
        critic2.gradient(x.state, qTarget);
    }
#else
    {
        std::size_t i = x.action.argmax();
        /* select action */
        const Tensor& nextProb = actor.forward(x.nextState);
        std::size_t k = nextProb.argmax();
        /* select value */;
        const Tensor& q1 = critic1Target.forward(x.nextState);
        const Tensor& q2 = critic2Target.forward(x.nextState);
        float q = std::min(q1[k], q2[k]);
        float nextValue = 0;
        nextValue = (nextProb[k] + beta)*(q - alpha[k]*std::log(nextProb[k] + beta));
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*nextValue;
        }
        const Tensor &out1 = critic1.forward(x.state);
        Tensor qTarget1 = out1;
        qTarget1[i] = qTarget;
        critic1.backward(Loss::MSE::df(out1, qTarget1));
        critic1.gradient(x.state, qTarget1);
        const Tensor &out2 = critic2.forward(x.state);
        Tensor qTarget2 = out2;
        qTarget2[i] = qTarget;
        critic2.backward(Loss::MSE::df(out2, qTarget2));
        critic2.gradient(x.state, qTarget2);
    }
#endif
    /* train policy net */
    {
        const Tensor& p = actor.forward(x.state);
        const Tensor& q1 = critic1.forward(x.state);
        const Tensor& q2 = critic2.forward(x.state);
        Tensor loss(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            float q = 0.5*(q1[i] + q2[i]);
            /*
                L(p) = p*(q - alpha*log(p))
                time shift loss: e^(beta*D)(L) = L(p + beta)
            */
            float dL = (p[i] + beta)*(q - alpha[i]*std::log(p[i] + beta));
            loss[i] = p[i] - dL;
        }
        actor.backward(loss);
        actor.gradient(x.state, loss);
    }

    /* alpha */
    {
        const Tensor& prob = x.action;
        for (int i = 0; i < actionDim; i++) {
            alpha.g[i] += (RL::entropy(prob[i]) - entropy0)*alpha[i];
        }
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
        critic1.softUpdateTo(critic1Target, 1e-3);
        critic2.softUpdateTo(critic2Target, 1e-3);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        float beta = float(k)/maxMemorySize;
        experienceReplay(memories[k], beta);
    }
    actor.RMSProp(1e-2, 0.9, 0);

#if 1
    std::cout<<"annealing:"<<annealing.val<<",alpha:";
    alpha.val.printValue();
#endif
    alpha.RMSProp(1e-5, 0.9, 0);
    critic1.RMSProp(1e-3, 0.9, 0);
    critic2.RMSProp(1e-3, 0.9, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size()/4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.3 ? 0.3 : exploringRate;
    learningSteps++;
    return;
}

void RL::SAC::save()
{
    actor.save("sac_actor");
    critic1.save("sac_critic1");
    critic2.save("sac_critic2");
    return;
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    critic1.load("sac_critic1");
    critic2.load("sac_critic2");
    critic1.copyTo(critic1Target);
    critic2.copyTo(critic2Target);
    return;
}
