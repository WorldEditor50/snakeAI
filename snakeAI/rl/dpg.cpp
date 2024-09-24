#include "dpg.h"
#include "layer.h"
#include "loss.h"
#include "attention.hpp"
#include "concat.hpp"

RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    //entropy0 = -0.11*std::log(0.11);
    entropy0 = -0.08*std::log(0.08);
#if 0
    policyNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
#else
    policyNet = Net(Attention<16>::_(stateDim, 4, true),
                    LayerNorm<Sigmoid, LN::Pre>::_(16*4, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
//    policyNet = Net(ScaledConcat<Layer<Sigmoid>, 16>::_(Layer<Sigmoid>(stateDim, 4, true, true), stateDim, 4, true),
//                   LayerNorm<Sigmoid, LN::Post>::_(16*4, hiddenDim, true, true),
//                   Layer<Softmax>::_(hiddenDim, actionDim, true, true));
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
    for (std::size_t t = 0; t < x.size(); t++) {
        const Tensor &prob = x[t].action;
        int k = x[t].action.argmax();
        alpha.g[k] += (-prob[k]*std::log(prob[k] + 1e-8) - entropy0)*alpha[t];
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state);
        policyNet.backward(Loss::CrossEntropy(out, x[t].action));
        policyNet.gradient(x[t].state, x[t].action);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.99999;
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
