#include "qlstm.h"

RL::QLSTM::QLSTM(std::size_t stateDim_, std::size_t hiddenDim_, std::size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    QMainNet = LSTM(stateDim_, hiddenDim_, actionDim_, true);
    QTargetNet = LSTM(stateDim_, hiddenDim_, actionDim_, false);
    this->QMainNet.copyTo(QTargetNet);
}

void RL::QLSTM::perceive(Vec& state,
        Vec& action,
        Vec& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Vec& RL::QLSTM::sample(Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    Vec &out = QMainNet.y;
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        out[index] = 1;
    } else {
        QMainNet.forward(state);
    }
    return out;
}

RL::Vec &RL::QLSTM::action(const Vec &state)
{
    return QMainNet.forward(state);
}

void RL::QLSTM::experienceReplay(const Transition& x, std::vector<Vec> &y)
{
    Vec qTarget(actionDim);
    /* estimate q-target: Q-Regression */
    /* select Action to estimate q-value */
    int i = RL::argmax(x.action);
    qTarget = QMainNet.forward(x.state);
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = RL::argmax(QMainNet.forward(x.nextState));
        /* select value in the QTargetNet */
        Vec &v = QTargetNet.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];
    }
    y.push_back(qTarget);
    return;
}

void RL::QLSTM::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    double learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        learningSteps = 0;
    }
    /* experience replay */
    std::vector<Vec> x;
    std::vector<Vec> y;
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Rand::engine);
        x.push_back(memories[k].state);
        experienceReplay(memories[k], y);
    }
    QMainNet.forward(x);
    QMainNet.gradient(x, y);
    QMainNet.Adam(learningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}
