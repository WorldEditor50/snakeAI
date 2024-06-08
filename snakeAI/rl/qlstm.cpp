#include "qlstm.h"
#include "lstm.h"
#include "layer.h"
#include "loss.h"

RL::QLSTM::QLSTM(std::size_t stateDim_, std::size_t hiddenDim_, std::size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    h = Tensor(hiddenDim_, 1);
    c = Tensor(hiddenDim_, 1);
    lstm = LSTM::_(stateDim_, hiddenDim_, hiddenDim_, true);
    QMainNet = Net(lstm,
                   TanhNorm<Sigmoid>::_(hiddenDim_, actionDim_, true));
    QTargetNet = Net(LSTM::_(stateDim_, hiddenDim_, hiddenDim_, false),
                     TanhNorm<Sigmoid>::_(hiddenDim_, actionDim_, false));
    QMainNet.copyTo(QTargetNet);
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
    Tensor& out = QMainNet.forward(state, true);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor& RL::QLSTM::noiseAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state, true);
    return noise(out, exploringRate);
}

RL::Tensor &RL::QLSTM::action(const Tensor &state)
{
    lstm->h = h;
    lstm->c = c;
    return QMainNet.forward(state, true);
}

void RL::QLSTM::reset()
{
    lstm->reset();
    return;
}

void RL::QLSTM::experienceReplay(const Transition& x)
{
    Tensor qTarget(actionDim, 1);
    /* estiTensore q-target: Q-Regression */
    /* select Action to estiTensore q-value */
    int i = x.action.argmax();
    Tensor out = QMainNet.forward(x.state, false);
    qTarget = out;
    if (x.done) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.forward(x.nextState, true).argmax();
        /* select value in the QTargetNet */
        Tensor &v = QTargetNet.forward(x.nextState, true);
        qTarget[i] = x.reward + gamma * v[k];
    }
    QMainNet.backward(Loss::MSE(out, qTarget));
    QMainNet.gradient(x.state, qTarget);
    return;
}

void RL::QLSTM::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{
    h = lstm->h;
    c = lstm->c;
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    lstm->reset();
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }
    QMainNet.RMSProp(learningRate, 0.9, 0);
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
