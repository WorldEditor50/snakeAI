#include "dpg.h"
ML::DPG::DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim)
{
    this->gamma = 0.9;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->policyNet = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim,
                              true, SIGMOID, CROSS_ENTROPY);
    return;
}

int ML::DPG::greedyAction(Vec &state)
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

int ML::DPG::randomAction()
{
    Vec& out = policyNet.getOutput();
    out.assign(actionDim, 0);
    int index = rand() % actionDim;
    out[index] = 1;
    return index;
}

int ML::DPG::action(Vec &state)
{
    int index = 0;
    policyNet.feedForward(state);
    Vec& out = policyNet.getOutput();
    index = maxAction(out);
    return index;
}

int ML::DPG::maxAction(Vec& value)
{
    int index = 0;
    double maxValue = value[0];
    for (std::size_t i = 0; i < value.size(); i++) {
        if (maxValue < value[i]) {
            maxValue = value[i];
            index = i;
        }
    }
    return index;
}

void ML::DPG::zscore(Vec &x)
{
    double u = 0;
    double n = 0;
    double sigma = 0;
    /* expectation */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        u += x[i];
        n++;
    }
    u = u / n;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] = x[i] - u;
        sigma += x[i] * x[i];
    }
    sigma = sqrt(sigma / n);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] = x[i] / sigma;
    }
    return;
}

void ML::DPG::reinforce(OptType optType, double learningRate, std::vector<Step>& x)
{
    double r = 0;
    double u = 0;
    double n = 0;
    Vec discoutedReward(x.size());
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discoutedReward[i] = r;
        u += r;
        n++;
    }
    u = u / n;
    for (std::size_t i = 0; i < x.size(); i++) {
        int k = maxAction(x[i].action);
        x[i].action[k] *= discoutedReward[i] - u;
        policyNet.gradient(x[i].state, x[i].action);
    }
    policyNet.optimize(optType, learningRate);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void ML::DPG::save(const std::string &fileName)
{
    policyNet.save(fileName);
    return;
}

void ML::DPG::load(const std::string &fileName)
{
    policyNet.load(fileName);
    return;
}
