#include "drpg.h"

RL::DRPG::DRPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim)
{
    this->gamma = 0.9;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->policyNet = LstmNet(LSTM(stateDim, hiddenDim, hiddenDim, true),
                              Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                              LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                              SoftmaxLayer::_(hiddenDim, actionDim, true));
}

RL::Mat &RL::DRPG::eGreedyAction(const Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Rand::engine);
    Mat &out = policyNet.output();
    if (p < exploringRate) {
        out.zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        out[index] = 1;
    } else {
        policyNet.forward(state);
    }
    return out;
}

RL::Mat &RL::DRPG::action(const Mat &state)
{
    return policyNet.forward(state);
}

void RL::DRPG::reinforce(const std::vector<Mat> &x, std::vector<Mat> &y, std::vector<float>& reward, float learningRate)
{
    float r = 0;
    for (int i = reward.size() - 1; i >= 0; i--) {
        r = gamma * r + reward[i];
        reward[i] = r;
    }
    Mat re(reward.size(), 1);
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
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}
