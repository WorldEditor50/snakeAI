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
    qNet2 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    qNetTarget1 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
    qNetTarget2 = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
    qNet1.copyTo(qNetTarget1);
    qNet2.copyTo(qNetTarget2);
}

RL::Tensor &RL::BCQ::action(const Tensor &state)
{
    return qNet1.forward(state);
}

void RL::BCQ::experienceReplay(const Transition& x)
{

    return;
}

void RL::BCQ::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{

    return;
}
