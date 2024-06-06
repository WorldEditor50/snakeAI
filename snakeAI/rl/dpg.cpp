#include "dpg.h"
#include "layer.h"
#include "loss.h"
#include "attention.hpp"

RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.05*std::log(0.05);
#if 0
    policyNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true),
                    Softmax::_(hiddenDim, actionDim, true));
#else
    policyNet = Net(ScaledDotProduct::_(stateDim, hiddenDim, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true),
                    Softmax::_(hiddenDim, actionDim, true));
#endif
}

RL::Tensor &RL::DPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::DPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return noise(out);
}

RL::Tensor &RL::DPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::DPG::action(const Tensor &state)
{
    return policyNet.forward(state);
}

void RL::DPG::reinforce(std::vector<Step>& x, float learningRate)
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
    alpha.RMSProp(0.9, 1e-4, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    //policyNet.optimize(OPT_NORMRMSPROP, 1e-2);
    policyNet.RMSProp(0.9, learningRate, 0.1);
    policyNet.clamp(-1, 1);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::save(const std::string &fileName)
{
    //policyNet.save(fileName);
    return;
}

void RL::DPG::load(const std::string &fileName)
{
    //policyNet.load(fileName);
    return;
}
