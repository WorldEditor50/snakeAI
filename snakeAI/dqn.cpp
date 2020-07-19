#include "dqn.h"
namespace ML {
    void DQN::CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
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
        this->QMainNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1);
        this->QTargetNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0);
        this->QMainNet.CopyTo(QTargetNet);
        return;
    }

    void DQN::Perceive(std::vector<double>& state,
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

    int DQN::GreedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = rand() % actionDim;
        } else {
            index = Action(state);
        }
        return index;
    }

    int DQN::Action(std::vector<double> &state)
    {
        int index = 0;
        QMainNet.FeedForward(state);
        std::vector<double>& Action = QMainNet.GetOutput();
        index = MaxQ(Action);
        return index;
    }

    int DQN::MaxQ(std::vector<double>& q_value)
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

    void DQN::ExperienceReplay(Transition& x)
    {
        std::vector<double> qTarget(actionDim);
        std::vector<double>& QTargetNetOutput = QTargetNet.GetOutput();
        std::vector<double>& QMainNetOutput = QMainNet.GetOutput();
        /* estimate q-target: Q-Regression */
        /* select Action to estimate q-value */
        int i = int(x.action);
        QMainNet.FeedForward(x.state);
        qTarget = QMainNetOutput;
        if (x.done == true) {
            qTarget[i] = x.reward;
        } else {
            /* select optimal Action in the QMainNet */
            QMainNet.FeedForward(x.nextState);
            int k = MaxQ(QMainNetOutput);
            /* select value in the QTargetNet */
            QTargetNet.FeedForward(x.nextState);
            qTarget[i] = x.reward + gamma * QTargetNetOutput[k];
        }
        /* train QMainNet */
        QMainNet.Gradient(x.state, qTarget);
        return;
    }

    void DQN::Learn(int optType, double learningRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            //QMainNet.CopyTo(QTargetNet);
            QMainNet.SoftUpdateTo(QTargetNet, 0.01);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            ExperienceReplay(memories[k]);
        }
        QMainNet.Optimize(optType, learningRate);
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

    void DQN::Save(const std::string &fileName)
    {
        QMainNet.Save(fileName);
        return;
    }

    void DQN::Load(const std::string &fileName)
    {
        QMainNet.Load(fileName);
        QMainNet.CopyTo(QTargetNet);
        return;
    }
}
