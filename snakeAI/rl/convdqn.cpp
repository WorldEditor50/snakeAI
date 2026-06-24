#include "convdqn.h"
#include "layer.h"
#include "conv2d.hpp"
#include "loss.h"

RL::ConvDQN::ConvDQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
    QMainNet = Net(Conv2d<Tanh>::_(1, 118, 118, 1, 5, 5, 1, true, true),

                   MaxPooling2d::_(1, 24, 24, 2, 2),
                   Conv2d<Tanh>::_(1, 12, 12, 4, 3, 3, 0, true, true),
                   MaxPooling2d::_(4, 4, 4, 2, 2),
                   Layer<Tanh>::_(4*2*2, hiddenDim, true, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(Conv2d<Tanh>::_(1, 118, 118, 1, 5, 5, 1, true, false),
                     MaxPooling2d::_(1, 24, 24, 2, 2),
                     Conv2d<Tanh>::_(1, 12, 12, 4, 3, 3, 0, true, false),
                     MaxPooling2d::_(4, 4, 4, 2, 2),
                     Layer<Tanh>::_(4*2*2, hiddenDim, true, false),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));

    QMainNet.copyTo(QTargetNet);
}

void RL::ConvDQN::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor& RL::ConvDQN::eGreedyAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor& RL::ConvDQN::noiseAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return noise(out, exploringRate);
}

RL::Tensor &RL::ConvDQN::action(const Tensor &state)
{
    return QMainNet.forward(state);
}

void RL::ConvDQN::experienceReplay(const Transition& x)
{
    /* — Step 1: compute next-state Q values for TD-target — */
    int i = x.action.argmax();
    int k = 0;
    float tdTarget = x.reward;

    if (!x.done) {
        /* use QMainNet to SELECT optimal next action (argmax) */
        Tensor& nextMainOut = QMainNet.forward(x.nextState);
        k = nextMainOut.argmax();
        /* use QTargetNet to EVALUATE next-state Q-value */
        Tensor& nextTargetOut = QTargetNet.forward(x.nextState);
        tdTarget = x.reward + gamma * nextTargetOut[k];
    }

    /* — Step 2: forward current state (restores correct activations) — */
    Tensor out = QMainNet.forward(x.state);
    Tensor qTarget = out;
    qTarget[i] = tdTarget;

    /* — Step 3: train QMainNet using TD-loss — */
    QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    return;
}

void RL::ConvDQN::learn(std::size_t maxMemorySize,
                    std::size_t /*replaceTargetIter*/,
                    std::size_t batchSize,
                    float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* Polyak soft-update target network (smooth & stable) */
    QMainNet.softUpdateTo(QTargetNet, 0.01);

    /* Adam optimizer for stable convergence */
    QMainNet.Adam(learningRate, 0.99, 0.9, 1e-4);

    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

