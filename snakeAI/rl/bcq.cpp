#include "bcq.h"
#include "layer.h"
#include "loss.h"

RL::BCQ::BCQ(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    featureDim = stateDim + actionDim;
    stateGenerator = VAE(stateDim, 2*stateDim, 2);
    actionGenerator = VAE(actionDim, 2*actionDim, 2);
    qNet1 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    qNet2 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
    qNet1.copyTo(qNet2);
}

RL::Tensor &RL::BCQ::action(const Tensor &state)
{
    return qNet1.forward(state);
}

void RL::BCQ::experienceReplay(const Transition& x)
{
    Tensor s = stateGenerator.forward(x.state);
    stateGenerator.backward(x.state);
    Tensor a = actionGenerator.forward(x.action);
    actionGenerator.backward(x.action);

    /* estiTensore q-target: Q-Regression */
    /* select Action to estiTensore q-value */
    int i = a.argmax();
    Tensor out = qNet1.forward(s);
    Tensor qTarget = out;
    if (x.done) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = qNet1.forward(x.nextState).argmax();
        /* select value in the QTargetNet */
        Tensor &v = qNet2.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];
    }
    /* train QMainNet */
    qNet1.backward(Loss::MSE(out, qTarget));
    qNet1.gradient(x.state, qTarget);
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
        std::cout<<"update target net"<<std::endl;
        /* update tagetNet */
        qNet1.softUpdateTo(qNet2, 0.01);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }
    qNet1.RMSProp(learningRate, 0.9, 0);
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
