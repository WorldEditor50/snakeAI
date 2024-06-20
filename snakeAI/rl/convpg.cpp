#include "convpg.h"
#include "layer.h"
#include "conv2d.hpp"
#include "loss.h"

RL::ConvPG::ConvPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.05*std::log(0.05);
    policyNet = Net(Conv2d<Tanh>::_(1, 118, 118, 4, 5, 5, 1, true, true),
                    MaxPooling2d::_(4, 24, 24, 2, 2),
                    Conv2d<Sigmoid>::_(4, 12, 12, 8, 3, 3, 0, true, true),
                    MaxPooling2d::_(8, 4, 4, 2, 2),
                    Layer<Sigmoid>::_(8*2*2, hiddenDim, true, true),
                    LayerNorm<Tanh, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
}

RL::Tensor &RL::ConvPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::ConvPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return noise(out);
}

RL::Tensor &RL::ConvPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::ConvPG::action(const Tensor &state)
{
    return policyNet.forward(state);
}

void RL::ConvPG::reinforce(std::vector<Step>& x, float learningRate)
{
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    for (std::size_t i = 0; i < x.size(); i++) {
        const Tensor &prob = x[i].action;
        int k = x[i].action.argmax();
        alpha.g[k] += -prob[k]*std::log(prob[k] + 1e-8) - entropy0;
        x[i].action[k] = prob[k]*(discountedReward[i] - u);
        Tensor &out = policyNet.forward(x[i].state);
        policyNet.backward(Loss::CrossEntropy(out, x[i].action));
        policyNet.gradient(x[i].state, x[i].action);
    }
    alpha.RMSProp(1e-4, 0.9, 0);
#if 0
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0.1);
    policyNet.clamp(-1, 1);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

