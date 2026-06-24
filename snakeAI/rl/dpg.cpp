#include "dpg.h"
#include "layer.h"
#include "loss.h"
#include "attention.hpp"
#include "transformer.hpp"
#include "moe.hpp"

RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.9), exploringRate(1)
{
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    H0 = RL::entropy(0.25);
#if 0
    policyNet = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
#else
    policyNet = Net(MOE<4, 8>::_(stateDim, true),
                    LayerNorm<Sigmoid, LN::Pre>::_(stateDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
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
        float H = RL::entropy(prob[k]);
        alpha.g[k] += H0 - H;
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state);
        Tensor dLoss = Loss::CrossEntropy::df(out, x[t].action);
        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-7, 0.9, 0);
    alpha.clamp(0.2, 0.2, 1);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::reinforce1(std::vector<Step>& x, float learningRate)
{
    /*
        Standard REINFORCE with baseline — clean, no side effects version.

        ∇J = E[∇log π(a|s) · (G_t - b)]

        For softmax policy π(a|s) = exp(z_a)/Σexp(z_i):
          ∂log π(a|s)/∂z_i = δ_{ik} - π_i = (e_k - π)_i
          ∇_z J = A · (e_k - π)

        We set dLoss such that the softmax Jacobian-vector product gives -∇J:
          J · dLoss = -A · (e_k - π)

        Setting dLoss[k] = -A/π_k, dLoss[i≠k] = 0:
          (J·dLoss)_i = y_i · (dLoss_i - Σ y_j·dLoss_j)
                      = y_i · (0 if i≠k else -A/π_k  -  y_k·(-A/π_k))
                      = y_i · (-δ_{ik}·A/π_k + A)
                      = -A · (δ_{ik} - y_i) = -A · (e_k - y)_i = -∇_i J ✓

        Then RMSProp does θ -= η·g.w = θ -= η·(-A·(e_k-π)·x^T)
        = θ += η·A·(e_k-π)·x^T = gradient ascent. ✓

        Note: dLoss[k] uses NEGATIVE advantage because RMSProp optimizer
        subtracts gradients from parameters. Without negation, we'd get
        gradient descent on the policy gradient objective.
    */
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    for (std::size_t t = 0; t < x.size(); t++) {
        const Tensor &oneHotAction = x[t].action;
        int k = oneHotAction.argmax();

        /* --- Forward pass to get current policy --- */
        Tensor &out = policyNet.forward(x[t].state);

        /* --- alpha (temperature) gradient ---
           Full distribution entropy H = -Σ π_i·log(π_i) */
        float H = RL::entropy(out[k]);
        alpha.g[k] += H0 - H;

        /* --- REINFORCE policy gradient (ascent via negated dLoss) --- */
        float advantage = discountedReward[t] - u;
        Tensor dLoss(actionDim, 1);
        dLoss[k] = -advantage / (out[k] + 1e-8f);   /* negative = gradient ascent */

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-7, 0.9, 0);
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
    //policyNet.load(fileName);
    return;
}
