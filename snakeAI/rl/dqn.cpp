#include "dqn.h"
RL::DQN::DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->QMainNet = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID);
    this->QTargetNet = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID);
    this->QMainNet.copyTo(QTargetNet);
    return;
}

void RL::DQN::perceive(Vec& state,
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

RL::Vec& RL::DQN::greedyAction(Vec &state)
{
    double p = double(rand() % 10000) / 10000;
    Vec &out = QMainNet.output();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        int index = rand() % actionDim;
        out[index] = 1;
    } else {
        QMainNet.feedForward(state);
    }
    return out;
}

int RL::DQN::action(Vec &state)
{
    return QMainNet.feedForward(state).argmax();
}

void RL::DQN::experienceReplay(Transition& x)
{
    Vec qTarget(actionDim);
    Vec& QTargetNetOutput = QTargetNet.output();
    Vec& QMainNetOutput = QMainNet.output();
    /* estimate q-target: Q-Regression */
    /* select Action to estimate q-value */
    int i = RL::argmax(x.action);
    QMainNet.feedForward(x.state);
    qTarget = QMainNetOutput;
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.feedForward(x.nextState).argmax();
        /* select value in the QTargetNet */
        QTargetNet.feedForward(x.nextState);
        qTarget[i] = x.reward + gamma * QTargetNetOutput[k];
    }
    /* train QMainNet */
    QMainNet.gradient(x.state, qTarget);
    return;
}

void RL::DQN::learn(OptType optType,
                    std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    double learningRate)
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
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        experienceReplay(memories[k]);
    }
    QMainNet.optimize(optType, learningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate = exploringRate * 0.99999;
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
