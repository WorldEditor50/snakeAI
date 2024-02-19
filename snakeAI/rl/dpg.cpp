#include "dpg.h"
RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    policyNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                     LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                     LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                     SoftmaxLayer::_(hiddenDim, actionDim, true));
}

RL::Mat &RL::DPG::eGreedyAction(const Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        policyNet.output().zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        policyNet.output()[index] = 1;
    } else {
        policyNet.forward(state);
    }
    return policyNet.output();
}

int RL::DPG::action(const Mat &state)
{
    int a = policyNet.forward(state).argmax();
    policyNet.show();
    return a;
}

void RL::DPG::reinforce(OptType optType, float learningRate, std::vector<Step>& x)
{
    float r = 0;
    Mat discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    for (std::size_t i = 0; i < x.size(); i++) {
        int k = x[i].action.argmax();
        float ri = discountedReward[i] - u;
        x[i].action[k] *= ri;
        policyNet.gradient(x[i].state, x[i].action, Loss::CrossEntropy);
    }
    policyNet.optimize(optType, learningRate, 0.1);
    policyNet.clamp(-1, 1);
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
