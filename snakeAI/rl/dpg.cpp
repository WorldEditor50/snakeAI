#include "dpg.h"
RL::DPG::DPG(std::size_t stateDim,
             std::size_t hiddenDim,
             std::size_t hiddenLayerNum,
             std::size_t actionDim)
{
    this->gamma = 0.9;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->policyNet = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim,
                              true, SIGMOID, CROSS_ENTROPY);
    return;
}

int RL::DPG::greedyAction(Vec &state)
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

int RL::DPG::randomAction()
{
    Vec& out = policyNet.output();
    out.assign(actionDim, 0);
    int index = rand() % actionDim;
    out[index] = 1;
    return index;
}

int RL::DPG::action(Vec &state)
{
    return policyNet.feedForward(state).argmax();
}

void RL::DPG::reinforce(OptType optType, double learningRate, std::vector<Step>& x)
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
        int k = RL::max(x[i].action);
        x[i].action[k] *= discoutedReward[i] - u;
        policyNet.gradient(x[i].state, x[i].action);
    }
    policyNet.optimize(optType, learningRate);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::save(const std::string &fileName)
{
    policyNet.save(fileName);
    return;
}

void RL::DPG::load(const std::string &fileName)
{
    policyNet.load(fileName);
    return;
}
