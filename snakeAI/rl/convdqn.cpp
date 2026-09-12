#include "convdqn.h"
#include "layer.h"
#include "conv2d.hpp"
#include "loss.h"

RL::ConvDQN::ConvDQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
    /*
       The Q-head used to be Layer<Sigmoid>, which confined every Q-value to
       (0, 1). Snake's reward is mostly negative: dying costs -1.5, and the
       distance-shaping term in Environment::reward0 returns RAW distance deltas
       reaching about -167. TD targets are therefore routinely negative, and the
       network structurally could not represent them.

       Measured with test/test_convdqn.cpp on a 4-way spatial bandit with +1/-1
       rewards, the old configuration reached 1/4 correct greedy actions (chance)
       and every Q-value was stuck at exactly 0.000: the -1 targets drove the
       sigmoid into saturation, where its derivative is ~0, so the gradient died
       and nothing could recover. A Q-head must be LINEAR (unbounded).

       The first convolution also emitted a SINGLE feature map at stride 5 — a
       severe bottleneck for a 118x118 board. It now emits 4 maps, and the second
       convolution 8 instead of 4.
    */
    QMainNet = Net(Conv2d<Tanh>::_(1, 118, 118, 4, 5, 5, 1, true, true),
                   MaxPooling2d::_(4, 24, 24, 2, 2),
                   Conv2d<Tanh>::_(4, 12, 12, 8, 3, 3, 0, true, true),
                   MaxPooling2d::_(8, 4, 4, 2, 2),
                   Layer<Tanh>::_(8*2*2, hiddenDim, true, true),
                   Layer<Linear>::_(hiddenDim, actionDim, true, true));

    QTargetNet = Net(Conv2d<Tanh>::_(1, 118, 118, 4, 5, 5, 1, true, false),
                     MaxPooling2d::_(4, 24, 24, 2, 2),
                     Conv2d<Tanh>::_(4, 12, 12, 8, 3, 3, 0, true, false),
                     MaxPooling2d::_(8, 4, 4, 2, 2),
                     Layer<Tanh>::_(8*2*2, hiddenDim, true, false),
                     Layer<Linear>::_(hiddenDim, actionDim, true, false));

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
    /* Value copy: eGreedy(hard = true) ZEROES the tensor it is given, and `out`
       is the network's own cached output buffer, which a later backward() pass
       needs. */
    qAction = out;
    /* hard = true: with a LINEAR Q-head the Q-values are unbounded, so merely
       overwriting one entry with 1 would not make it the argmax of anything
       (some other action could easily hold 12.0). Zeroing first makes the
       exploration choice unconditional, which is exactly what epsilon-greedy
       requires. */
    return eGreedy(qAction, exploringRate, true);
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

    /* Polyak soft-update target network (smooth & stable). Note that
       `replaceTargetIter` is accepted but NOT used: the target network is
       blended continuously instead of being hard-copied every N steps. */
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

    /*
       Exploration schedule.

       This used to decay by 0.99999 per learn() call, which needs
       ln(0.1)/ln(0.99999) ~ 230,000 calls to reach the 0.1 floor. The agent
       calls learn() once per game step, so epsilon effectively stayed at ~1.0
       for a whole session — measured 1.0 -> 0.67 after 40,000 calls — and
       ConvDQN therefore acted almost uniformly at random the entire time. That
       is the second reason it showed "no noticeable effect".

       0.9995 reaches the floor in ~6,000 calls (tens of episodes), which is a
       realistic exploration budget, and the floor is lowered to 0.05.
    */
    exploringRate *= 0.9995f;
    exploringRate = exploringRate < 0.05f ? 0.05f : exploringRate;
    learningSteps++;
    return;
}

