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
        this->policyNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
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
            index = randomAction();
        } else {
            index = action(state);
        }
        return index;
    }

    int DPGNet::randomAction()
    {
        std::vector<double>& policyNetOutput = policyNet.getOutput();
        policyNetOutput.assign(actionDim, 0);
        int index = rand() % actionDim;
        policyNetOutput[index] = 1;
        return index;
    }

    int DPGNet::action(std::vector<double> &state)
    {
        int index = 0;
        policyNet.feedForward(state);
        std::vector<double>& action = policyNet.getOutput();
        index = maxAction(action);
        return index;
    }

    int DPGNet::maxAction(std::vector<double>& value)
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

    void DPGNet::zscore(std::vector<double> &x)
    {
        double u = 0;
        double n = 0;
        double sigma = 0;
        for (int i = 0 ; i < x.size(); i++) {
            u += x[i];
            n++;
        }
        u = u / n;
        for (int i = 0 ; i < x.size(); i++) {
            x[i] = x[i] - u;
            sigma += x[i] * x[i];
        }
        sigma = sqrt(sigma / n);
        for (int i = 0 ; i < x.size(); i++) {
            x[i] = x[i] / sigma;
        }
        return;
    }

    void DPGNet::reinforce(std::vector<Step>& x)
    {
        double r = 0;
        std::vector<double> discoutedReward(x.size());
        for (int i = x.size() - 1; i >= 0; i--) {
            r = gamma * r + x[i].reward;
            discoutedReward[i] = r;
        }
        //zscore(discoutedReward);
        for (int i = 0; i < x.size(); i++) { 
            int k = maxAction(x[i].action);
            x[i].action[k] *= discoutedReward[i];
            policyNet.calculateGradient(x[i].state, x[i].action);
        }
        policyNet.RMSProp(0.9, learningRate);
        //policyNet.Adam(0.9, 0.99, 0.1);
        exploringRate *= 0.9999;
        exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
        return;
    }

    void DPGNet::save(const std::string &fileName)
    {
        policyNet.save(fileName);
        return;
    }

    void DPGNet::load(const std::string &fileName)
    {
        policyNet.load(fileName);
        return;
    }
}
