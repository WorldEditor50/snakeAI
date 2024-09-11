#include "drpg.h"
#include "layer.h"
#include "loss.h"

RL::DRPG::DRPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.02*std::log(0.02);
    lstm = LSTM::_(stateDim, hiddenDim, hiddenDim, true);
    h = Tensor(hiddenDim, 1);
    c = Tensor(hiddenDim, 1);
    policyNet = Net(lstm,
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
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
    lstm->h = h;
    lstm->c = c;
    return policyNet.forward(state, true);
}

void RL::DRPG::reinforce(std::vector<Step>& x, float learningRate)
{
    h = lstm->h;
    c = lstm->c;
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    lstm->reset();
    for (std::size_t t = 0; t < x.size(); t++) {
        const Tensor &prob = x[t].action;
        int k = x[t].action.argmax();
        alpha.g[k] += (-prob[k]*std::log(prob[k] + 1e-8) - entropy0)*alpha[t];
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state, false);
        policyNet.backward(Loss::CrossEntropy(out, x[t].action));
        policyNet.gradient(x[t].state, x[t].action);
    }
    alpha.RMSProp(1e-4, 0.9, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    return;
}
