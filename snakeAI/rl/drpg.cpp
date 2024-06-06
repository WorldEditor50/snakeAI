#include "drpg.h"
#include "layer.h"
#include "loss.h"

RL::DRPG::DRPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    lstm = LSTM::_(stateDim, hiddenDim, hiddenDim, true);
    policyNet = Net(lstm,
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                    LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                    Softmax::_(hiddenDim, actionDim, true));
}

RL::Tensor &RL::DRPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::DRPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return noise(out);
}

RL::Tensor &RL::DRPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return gumbelSoftmax(out, exploringRate);
}

RL::Tensor &RL::DRPG::action(const Tensor &state)
{
    return policyNet.forward(state, true);
}

void RL::DRPG::reinforce(std::vector<Step>& x, float learningRate)
{
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    lstm->reset();
    for (std::size_t i = 0; i < x.size(); i++) {
        const Tensor &prob = x[i].action;
        int k = x[i].action.argmax();
        x[i].action[k] = prob[k]*(discountedReward[i] - u);
        Tensor &out = policyNet.forward(x[i].state, false);
        policyNet.backward(Loss::CrossEntropy(out, x[i].action));
        policyNet.gradient(x[i].state, x[i].action);
    }
    policyNet.RMSProp(0.9, learningRate, 0.1);
    policyNet.clamp(-1, 1);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    return;
}
