#include "dpg.h"
namespace ML {
    void DPG::CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                          float learningRate)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
            return;
        }
        this->gamma = 0.9;
        this->baseLine = 1;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->learningRate = learningRate;
        this->policyNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
        return;
    }

    int DPG::GreedyAction(std::vector<float> &state)
    {
        if (state.size() != stateDim) {
            return -1;
        }
        float p = float(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = RandomAction();
        } else {
            index = Action(state);
        }
        return index;
    }

    int DPG::RandomAction()
    {
        std::vector<float>& policyNetOutput = policyNet.GetOutput();
        policyNetOutput.assign(actionDim, 0);
        int index = rand() % actionDim;
        policyNetOutput[index] = 1;
        return index;
    }

    int DPG::Action(std::vector<float> &state)
    {
        int index = 0;
        policyNet.FeedForward(state);
        std::vector<float>& Action = policyNet.GetOutput();
        index = maxAction(Action);
        return index;
    }

    int DPG::maxAction(std::vector<float>& value)
    {
        int index = 0;
        float maxValue = value[0];
        for (int i = 0; i < value.size(); i++) {
            if (maxValue < value[i]) {
                maxValue = value[i];
                index = i;
            }
        }
        return index;
    }

    void DPG::zscore(std::vector<float> &x)
    {
        float u = 0;
        float n = 0;
        float sigma = 0;
        /* expectation */
        for (int i = 0 ; i < x.size(); i++) {
            u += x[i];
            n++;
        }
        u = u / n;
        /* sigma */
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

    void DPG::reinforce(std::vector<Step>& x)
    {
        float r = 0;
        std::vector<float> discoutedReward(x.size());
        for (int i = x.size() - 1; i >= 0; i--) {
            r = gamma * r + x[i].reward;
            discoutedReward[i] = r;
        }
        //zscore(discoutedReward);
        for (int i = 0; i < x.size(); i++) {
            int k = maxAction(x[i].action);
            x[i].action[k] *= discoutedReward[i] - baseLine;
            policyNet.Gradient(x[i].state, x[i].action);
        }
        policyNet.RMSPropWithClip(0.9, 0.5, 2);
        //policyNet.Adam(0.9, 0.99, 0.5);
        exploringRate *= 0.9999;
        exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
        return;
    }

    void DPG::Save(const std::string &fileName)
    {
        policyNet.Save(fileName);
        return;
    }

    void DPG::Load(const std::string &fileName)
    {
        policyNet.Load(fileName);
        return;
    }
}
