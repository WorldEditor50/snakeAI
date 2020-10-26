#include "dqn.h"
ML::DQN::DQN(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim)
{
    if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
        return;
    }
    this->gamma = 0.99;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->QMainNet = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1);
    this->QTargetNet = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0);
    this->QMainNet.copyTo(QTargetNet);
    return;
}

void ML::DQN::perceive(std::vector<double>& state,
        std::vector<double>& action,
        std::vector<double>& nextState,
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

std::vector<double>& ML::DQN::greedyAction(std::vector<double> &state)
{
    double p = double(rand() % 10000) / 10000;
    std::vector<double> &out = QMainNet.getOutput();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        int index = rand() % actionDim;
        out[index] = 1;
    } else {
        QMainNet.feedForward(state);
    }
    return out;
}

int ML::DQN::action(std::vector<double> &state)
{
    int index = 0;
    QMainNet.feedForward(state);
    std::vector<double>& Action = QMainNet.getOutput();
    index = maxQ(Action);
    return index;
}

int ML::DQN::maxQ(std::vector<double>& q_value)
{
    int index = 0;
    double maxValue = q_value[0];
    for (int i = 0; i < q_value.size(); i++) {
        if (maxValue < q_value[i]) {
            maxValue = q_value[i];
            index = i;
        }
    }
    return index;
}

void ML::DQN::experienceReplay(Transition& x)
{
    std::vector<double> qTarget(actionDim);
    std::vector<double>& QTargetNetOutput = QTargetNet.getOutput();
    std::vector<double>& QMainNetOutput = QMainNet.getOutput();
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

void ML::DQN::learn(int optType,
                    int maxMemorySize,
                    int replaceTargetIter,
                    int batchSize,
                    double learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update tagetNet */
        QMainNet.copyTo(QTargetNet);
        learningSteps = 0;
    }
    /* experience replay */
    for (int i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        experienceReplay(memories[k]);
    }
    QMainNet.optimize(optType, learningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        int k = memories.size() / 3;
        for (int i = 0; i < k; i++) {
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
