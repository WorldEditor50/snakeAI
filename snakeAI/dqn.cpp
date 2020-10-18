#include "dqn.h"
namespace ML {
    void DQN::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
            return;
        }
        this->gamma = 0.99;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        this->QMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1);
        this->QTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0);
        this->QMainNet.copyTo(QTargetNet);
        return;
    }

    void DQN::perceive(std::vector<double>& state,
            double action,
            std::vector<double>& nextState,
            double reward,
            bool done)
    {
        if (state.size() != stateDim || nextState.size() != stateDim) {
            return;
        }
        Transition t(state, action, nextState, reward, done);
        memories.push_back(t);
        return;
    }

    int DQN::greedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = rand() % actionDim;
        } else {
            index = action(state);
        }
        return index;
    }

    int DQN::action(std::vector<double> &state)
    {
        int index = 0;
        QMainNet.feedForward(state);
        std::vector<double>& Action = QMainNet.getOutput();
        index = maxQ(Action);
        return index;
    }

    int DQN::maxQ(std::vector<double>& q_value)
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

    void DQN::experienceReplay(Transition& x)
    {
        std::vector<double> qTarget(actionDim);
        std::vector<double>& QTargetNetOutput = QTargetNet.getOutput();
        std::vector<double>& QMainNetOutput = QMainNet.getOutput();
        /* estimate q-target: Q-Regression */
        /* select Action to estimate q-value */
        int i = int(x.action);
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

    void DQN::learn(int optType, double learningRate)
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

    void DQN::save(const std::string &fileName)
    {
        QMainNet.save(fileName);
        return;
    }

    void DQN::load(const std::string &fileName)
    {
        QMainNet.load(fileName);
        QMainNet.copyTo(QTargetNet);
        return;
    }
}
