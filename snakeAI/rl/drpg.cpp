#include "drpg.h"

RL::DRPG::DRPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    policyNet = LstmNet(LSTM(stateDim, hiddenDim, hiddenDim, true),
                        Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                        LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true),
                        Softmax::_(hiddenDim, actionDim, true));
}

RL::Tensor &RL::DRPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::DRPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return noise(out);
}

RL::Tensor &RL::DRPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return gumbelSoftmax(out, exploringRate);
}

RL::Tensor &RL::DRPG::action(const Tensor &state)
{
    return policyNet.forward(state);
}

void RL::DRPG::reinforce(const std::vector<Tensor> &x, std::vector<Tensor> &y, std::vector<float>& reward, float learningRate)
{
    float r = 0;
    for (int i = reward.size() - 1; i >= 0; i--) {
        r = gamma * r + reward[i];
        reward[i] = r;
    }
    Tensor re(reward.size(), 1);
    re.val = reward;
    float u = re.mean();
    for (std::size_t t = 0; t < y.size(); t++) {
        int k = y[t].argmax();
        y[t][k] *= reward[t] - u;
    }
    policyNet.reset();
    policyNet.forward(x);
    policyNet.backward(x, y, Loss::CrossEntropy);
    policyNet.optimize(learningRate, 0.1);
    policyNet.clamp(-1, 1);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    return;
}
