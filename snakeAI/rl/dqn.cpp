#include "dqn.h"
#include "layer.h"
#include "loss.h"
#include "attention.hpp"

RL::DQN::DQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    totalReward = 0;
    stateDim = stateDim_;
    actionDim = actionDim_;

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
    /* estiTensore q-target: Q-Regression */
    /* select Action to estiTensore q-value */
    int i = x.action.argmax();
    Tensor out = QMainNet.forward(x.state);
    Tensor qTarget = out;
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.forward(x.nextState).argmax();
        /* select value in the QTargetNet */
        Tensor &v = QTargetNet.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];
    }
    /* train QMainNet */
    QMainNet.backward(Loss::MSE(out, qTarget));
    QMainNet.gradient(x.state, qTarget);
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
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update tagetNet */
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        //QMainNet.copyTo(QTargetNet);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }
    QMainNet.RMSProp(learningRate, 0.9, 0);
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

void RL::DQN::save(const std::string &fileName)
{
    //QMainNet.save(fileName);
    return;
}

void RL::DQN::load(const std::string &fileName)
{
    //QMainNet.load(fileName);
    //QMainNet.copyTo(QTargetNet);
    return;
}
