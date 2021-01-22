#include "dqn.h"
ML::DQN::DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->QMainNet = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID);
    this->QTargetNet = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID);
    this->QMainNet.copyTo(QTargetNet);
    return;
}

void ML::DQN::perceive(Vec& state,
        Vec& action,
        Vec& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    ;
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

ML::Vec& ML::DQN::greedyAction(Vec &state)
{
    double p = double(rand() % 10000) / 10000;
    Vec &out = QMainNet.getOutput();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        int index = rand() % actionDim;
        out[index] = 1;
    } else {
        QMainNet.feedForward(state);
    }
    return out;
}

int ML::DQN::action(Vec &state)
{
    int index = 0;
    QMainNet.feedForward(state);
    Vec& Action = QMainNet.getOutput();
    index = maxQ(Action);
    return index;
}

int ML::DQN::maxQ(Vec& q_value)
{
    int index = 0;
    double maxValue = q_value[0];
    for (std::size_t i = 0; i < q_value.size(); i++) {
        if (maxValue < q_value[i]) {
            maxValue = q_value[i];
            index = i;
        }
    }
    return index;
}

void ML::DQN::experienceReplay(Transition& x)
{
    Vec qTarget(actionDim);
    Vec& QTargetNetOutput = QTargetNet.getOutput();
    Vec& QMainNetOutput = QMainNet.getOutput();
    /* estimate q-target: Q-Regression */
    /* select Action to estimate q-value */
    int i = maxQ(x.action);
    QMainNet.feedForward(x.state);
    qTarget = QMainNetOutput;
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        QMainNet.feedForward(x.nextState);
        int k = maxQ(QMainNetOutput);
        /* select value in the QTargetNet */
        QTargetNet.feedForward(x.nextState);
        qTarget[i] = x.reward + gamma * QTargetNetOutput[k];
    }
    /* train QMainNet */
    QMainNet.gradient(x.state, qTarget);
    return;
}

void ML::DQN::learn(OptType optType,
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

void ML::DQN::save(const std::string &fileName)
{
    QMainNet.save(fileName);
    return;
}

void ML::DQN::load(const std::string &fileName)
{
    QMainNet.load(fileName);
    QMainNet.copyTo(QTargetNet);
    return;
}
