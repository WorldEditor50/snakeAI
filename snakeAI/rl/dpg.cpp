#include "dpg.h"
RL::DPG::DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim)
{
    this->gamma = 0.9;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->policyNet = BPNN(SwishLayer::_(stateDim, hiddenDim, true),
                           DropoutLayer<Tanh>::_(hiddenDim, hiddenDim, true, 0.5),
                           LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                           SwishLayer::_(hiddenDim, hiddenDim, true),
                           DropoutLayer<Tanh>::_(hiddenDim, hiddenDim, true, 0.5),
                           LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                           SoftmaxLayer::_(hiddenDim, actionDim, true));
}

RL::Vec &RL::DPG::eGreedyAction(const Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        policyNet.output().assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        policyNet.output()[index] = 1;
    } else {
        policyNet.feedForward(state);
    }
    return policyNet.output();
}

int RL::DPG::action(const Vec &state)
{
    return policyNet.show(), policyNet.feedForward(state).argmax();
}

void RL::DPG::reinforce(OptType optType, double learningRate, std::vector<Step>& x)
{
    double r = 0;
    Vec discoutedReward(x.size(), 0);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discoutedReward[i] = r;
    }
    double u = RL::mean(discoutedReward);
    for (std::size_t i = 0; i < x.size(); i++) {
        int k = RL::argmax(x[i].action);
        x[i].action[k] *= (discoutedReward[i] - u);
        policyNet.gradient(x[i].state, x[i].action, Loss::CROSS_EMTROPY);
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
