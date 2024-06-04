#include "qlstm.h"

RL::QLSTM::QLSTM(std::size_t stateDim_, std::size_t hiddenDim_, std::size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    QMainNet = LstmNet(LSTM(stateDim_, hiddenDim_, hiddenDim_, true),
                       TanhNorm<Sigmoid>::_(hiddenDim_, actionDim_, true));
    QTargetNet = LstmNet(LSTM(stateDim_, hiddenDim_, hiddenDim_, false),
                         TanhNorm<Sigmoid>::_(hiddenDim_, actionDim_, false));
    this->QMainNet.copyTo(QTargetNet);
}

void RL::QLSTM::perceive(Tensor& state,
        Tensor& action,
        Tensor& nextState,
        float reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor& RL::QLSTM::eGreedyAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::QLSTM::action(const Tensor &state)
{
    return QMainNet.forward(state);
}

void RL::QLSTM::reset()
{
    QMainNet.reset();
    return;
}

void RL::QLSTM::experienceReplay(const Transition& x, std::vector<Tensor> &y)
{
    Tensor qTarget(actionDim, 1);
    /* estiTensore q-target: Q-Regression */
    /* select Action to estiTensore q-value */
    int i = x.action.argmax();
    qTarget = QMainNet.forward(x.state);
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.forward(x.nextState).argmax();
        /* select value in the QTargetNet */
        Tensor &v = QTargetNet.forward(x.nextState);
        qTarget[i] = x.reward + gamma * v[k];
    }
    y.push_back(qTarget);
    return;
}

void RL::QLSTM::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
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
    std::vector<Tensor> x;
    std::vector<Tensor> y;
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        x.push_back(memories[k].state);
        experienceReplay(memories[k], y);
    }
    QMainNet.forward(x);
    QMainNet.backward(x, y, Loss::MSE);
    QMainNet.optimize(learningRate, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
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
