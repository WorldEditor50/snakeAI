#include "policyGradient.h"
namespace ML {
    void DPGNet::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                          double learningRate)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
            return;
        }
        this->gamma = 0.9;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->learningRate = learningRate;
        this->policyNet.createNetWithSoftmax(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID);
        return;
    }

    int DPGNet::eGreedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            std::vector<double>& policyNetOutput = policyNet.getOutput();
            for (int i = 0; i < policyNetOutput.size(); i++) {
                policyNetOutput[i] = double(rand() % 100000) / 100000;
            }
            index = maxQ(policyNetOutput);
        } else {
            index = action(state);
        }
        return index;
    }

    int DPGNet::action(std::vector<double> &state)
    {
        int index = 0;
        policyNet.feedForward(state);
        std::vector<double>& action = policyNet.getOutput();
        index = maxQ(action);
        return index;
    }

    int DPGNet::maxQ(std::vector<double>& value)
    {
        int index = 0;
        double maxValue = value[0];
        for (int i = 0; i < value.size(); i++) {
            if (maxValue < value[i]) {
                maxValue = value[i];
                index = i;
            }
        }
        return index;
    }

    void DPGNet::normalize(std::vector<double> &x)
    {
        double avg = 0;
        double n = 0;
        double s = 0;
        for (int i = 0 ; i < x.size(); i++) {
            avg += x[i];
            n++;
        }
        avg = avg / n;
        for (int i = 0 ; i < x.size(); i++) {
            s += (x[i] - avg) * (x[i] - avg);
        }
        s = sqrt(s / n);
        for (int i = 0 ; i < x.size(); i++) {
            s += (x[i] - avg) * (x[i] - avg);
            x[i] = (x[i] - avg) / s;
        }
        return;
    }

    void DPGNet::reinforce(std::vector<Step>& x)
    {
        double r = 0;
        std::vector<double> discountedReward(x.size());
        for (int i = x.size() - 1; i >= 0; i--) {
            r = gamma * r + x[i].reward;
            discountedReward[i] = r;
        }
        for (int i = 0; i < x.size(); i++) {
            for (int j = 0; j < actionDim; j++) {
                x[i].action[j] *= discountedReward[i];
            }
            policyNet.calculateBatchGradient(x[i].state, x[i].action);
        }
        /* gradient ascent */
        //policyNet.Adam(0.9, 0.99, learningRate);
        policyNet.RMSProp(0.9, learningRate);
        exploringRate *= 0.95;
        if (exploringRate < 0.4) {
            exploringRate = 0.4;
        }
        return;
    }

    void DPGNet::save(const std::string &fileName)
    {
        policyNet.saveParameter(fileName);
        return;
    }

    void DPGNet::load(const std::string &fileName)
    {
        policyNet.loadParameter(fileName);
        return;
    }
}
