#include "dqn.h"
#include "layer.h"
#include "loss.h"
#include "concat.hpp"
#include "attention.hpp"
#include "transformer.hpp"
#include "moe.hpp"

RL::DQN::DQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
#if 0
    QMainNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                   TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                     TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
#elif 1
    /* it still performs well if stop exploring enviroment */
    QMainNet = Net(ScaledConcat<Layer<Sigmoid>, 16>::_(Layer<Sigmoid>(stateDim, 4, true, true), stateDim, 4, true),
                   TanhNorm<Sigmoid>::_(16*4, hiddenDim, true, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(ScaledConcat<Layer<Sigmoid>, 16>::_(Layer<Sigmoid>(stateDim, 4, true, false), stateDim, 4, false),
                     TanhNorm<Sigmoid>::_(16*4, hiddenDim, true, false),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
#else
    QMainNet = Net(MOE<8, 4>::_(stateDim, true),
                   TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(MOE<8, 4>::_(stateDim, false),
                     TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, false),
                     Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
#endif
    QMainNet.copyTo(QTargetNet);
}

void RL::DQN::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor& RL::DQN::eGreedyAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor& RL::DQN::noiseAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return noise(out, exploringRate);
}

RL::Tensor &RL::DQN::action(const Tensor &state)
{
    return QMainNet.forward(state);
}

void RL::DQN::experienceReplay(const Transition& x)
{
    /*
     * IMPORTANT: Compute Q-target BEFORE the QMainNet.backward() call.
     *
     * CRITICAL BUG FIX: We must save QMainNet.forward(x.state) results
     * BEFORE calling QMainNet.forward(x.nextState), because forward()
     * overwrites the network's internal cached o-values. If we call
     * forward(x.nextState) first, then backward(x.state, ...) uses the
     * wrong cached intermediate values, corrupting all gradients.
     *
     * The safe ordering is:
     *   1. Forward x.state → out (deep copy)
     *   2. Compute qTarget from out (already have action index i)
     *   3. Call forward(x.nextState) only on QTargetNet (separate network)
     *   4. Backward on QMainNet with x.state → uses correctly cached state
     */
    int i = x.action.argmax();

    if (x.done) {
        /* Terminal state: Q-target = reward directly */
        Tensor out = QMainNet.forward(x.state);
        Tensor qTarget = out;
        qTarget[i] = x.reward;
        QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    } else {
        /*
         * Non-terminal: need max_a' Q(s',a')
         * Use QTargetNet to evaluate, QMainNet to select action.
         * QMainNet.forward(x.state) saves the cached forward for backward.
         * CRITICAL: Don't call QMainNet.forward(x.nextState) BEFORE backward!
         */
        /* Forward on x.state first (caches values for backward) */
        Tensor out = QMainNet.forward(x.state);
        Tensor qTarget = out;

        /* Select best next action using QMainNet on SEPARATE network copy
           to avoid overwriting the cached x.state forward in QMainNet */
        int k = QTargetNet.forward(x.nextState).argmax();

        /* Evaluate next-state value using QTargetNet */
        Tensor &v = QTargetNet.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];

        /* Backward on QMainNet with x.state (cached values intact) */
        QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    }
    return;
}

void RL::DQN::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    /* update target network periodically (Polyak soft update) */
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        learningSteps = 0;
    }

    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* apply optimizer */
    QMainNet.RMSProp(learningRate, 0.9, 0);

    /* manage replay buffer: drop oldest entries when full */
    if (memories.size() > maxMemorySize + batchSize) {
        std::size_t k = std::min(batchSize, memories.size() - maxMemorySize);
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }

    /* decay exploring rate: 0.9999^n: 1→0.5 at ~6931 steps, 1→0.1 at ~23026 steps */
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::DQN::save(const std::string &fileName)
{
    QMainNet.save(fileName);
    return;
}

void RL::DQN::load(const std::string &fileName)
{
    QMainNet.load(fileName);
    QMainNet.copyTo(QTargetNet);
    return;
}
