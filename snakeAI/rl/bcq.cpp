#include "bcq.h"
#include "layer.h"
#include "loss.h"

RL::BCQ::BCQ(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99)
{
    featureDim = stateDim + actionDim;
    encoder = VAE(featureDim, 2*featureDim, stateDim);
    actor = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));
    for (int i = 0; i < max_qnet_num; i++) {
        critics[i] = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, true),
                         TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                         Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));
        criticsTarget[i] = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, false),
                               TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                               Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
        critics[i].copyTo(criticsTarget[i]);
    }
}

RL::Tensor &RL::BCQ::action(const Tensor &state)
{
    Tensor ga = encoder.decode(state);
    Tensor sa = Tensor::concat(1, state, ga);
    return actor.forward(sa);
}

RL::Tensor &RL::BCQ::mixAction(const RL::Tensor &state, const RL::Tensor &ga)
{
    Tensor sa = Tensor::concat(1, state, ga);
    Tensor& a = actor.forward(sa);
    a += ga;
    return a;
}

void RL::BCQ::perceive(const RL::Tensor &state,
                       const RL::Tensor &action,
                       const RL::Tensor &nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void RL::BCQ::experienceReplay(const Transition& x)
{
    /* train encoder */
    {
        Tensor sa = Tensor::concat(1, x.state, x.action);
        encoder.forward(sa);
        encoder.backward(sa);
    }

    /* train critics */
    {
        /* select action */
        Tensor ga = encoder.decode(x.nextState);
        Tensor san = Tensor::concat(1, x.nextState, ga);
        const Tensor& nextProb = actor.forward(san);
        std::size_t k = nextProb.argmax();
        /* select value */;
        float q = 0;
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = criticsTarget[i].forward(san);
            q += qi[k];
        }
        q /= max_qnet_num;
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*q;
        }
        Tensor sa = Tensor::concat(1, x.state, x.action);
        k = x.action.argmax();
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor &out = critics[i].forward(sa);
            Tensor p = out;
            p[k] = qTarget;
            critics[i].backward(Loss::MSE::df(out, p));
            critics[i].gradient(x.state, p);
        }
    }
    /* train actor */
    {
        Tensor ga = encoder.decode(x.state);
        Tensor sa = Tensor::concat(1, x.state, ga);
        const Tensor& p = actor.forward(sa);
        Tensor loss(actionDim, 1);
        Tensor q(actionDim, 1);
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = critics[i].forward(x.state);
            q += qi;
        }
        q /= max_qnet_num;
        for (int i = 0; i < actionDim; i++) {
            loss[i] = p[i] - q[i];
        }
        actor.backward(loss);
        actor.gradient(sa, loss);
    }
    return;
}

void RL::BCQ::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        /* update */
        for (int i = 0; i < max_qnet_num; i++) {
            critics[i].softUpdateTo(criticsTarget[i], (i + 1)*1e-4);
        }
        learningSteps = 0;
    }
    learningSteps++;
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }
    actor.RMSProp(1e-2, 0.9, 0);

    for (int i = 0; i < max_qnet_num; i++) {
        critics[i].RMSProp(1e-3, 0.9, 0);
    }

    return;
}
