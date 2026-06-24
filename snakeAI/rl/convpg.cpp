#include "convpg.h"
#include "layer.h"
#include "conv2d.hpp"
#include "loss.h"

RL::ConvPG::ConvPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.9), exploringRate(1)
{
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.05*std::log(0.05);
    policyNet = Net(Conv2d<Tanh>::_(1, 118, 118, 4, 5, 5, 1, true, true),
                    MaxPooling2d::_(4, 24, 24, 2, 2),
                    Conv2d<Tanh>::_(4, 12, 12, 8, 3, 3, 0, true, true),
                    MaxPooling2d::_(8, 4, 4, 2, 2),
                    Layer<Tanh>::_(8*2*2, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
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

    /* --- Standard REINFORCE policy gradient ---
       ∇J = ∇log π(a|s) · A

       For a softmax policy π(a|s) = exp(z_a)/Σexp(z_i):
         ∂log π(a|s)/∂z_k = 1 - π(a|s) if a=k,  -π(k|s) if a≠k

       The gradient w.r.t. logits: ∂J/∂z = A · (e_a - π(·|s))

       Through the softmax Jacobian J (where J_ij = π_i(δ_ij - π_j)):
         J · e = A · (e_a - π)

       Setting e_a = A/π(a|s), e_i≠a = 0 gives the correct result:
         (J · e)_a = π_a·(A/π_a) - π_a·A = A·(1-π_a) ✓
         (J · e)_i = 0 - π_i·A = -π_i·A ✓

       where π(a|s) = out[k] is the current policy probability of action a.
    */
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
        alpha.g[k] += (RL::entropy(prob[k]) - entropy0)*alpha[k];
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state);
        Tensor dLoss = Loss::CrossEntropy::df(out, x[t].action);
        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-4, 0.9, 0);
#if 0
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

