#include "mpg.h"
#include "layer.h"
#include "loss.h"

namespace RL {

MPG::MPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.9), exploringRate(1)
{
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    H0 = RL::entropy(0.25);

    /*
     * MambaLayer: input stateDim → hiddenDim internal → hiddenDim output (tanh)
     * The Mamba provides temporal credit assignment via its selective SSM state.
     */
    mamba = MambaLayer::_(stateDim, hiddenDim, hiddenDim, true);
    mamba_h = Tensor(hiddenDim, 1);

    policyNet = Net(mamba,
                    LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                    Layer<Softmax>::_(hiddenDim, actionDim, true, true));
}

Tensor &MPG::eGreedyAction(const Tensor &state)
{
    mamba->h = mamba_h;
    Tensor& out = policyNet.forward(state, true);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &MPG::noiseAction(const RL::Tensor &state)
{
    mamba->h = mamba_h;
    Tensor& out = policyNet.forward(state, true);
    return noise(out);
}

RL::Tensor &MPG::gumbelMax(const RL::Tensor &state)
{
    mamba->h = mamba_h;
    Tensor& out = policyNet.forward(state, true);
    return gumbelSoftmax(out, alpha.val);
}

Tensor &MPG::action(const Tensor &state)
{
    /* Restore persistent Mamba hidden state for inference/trajectory continuation */
    mamba->h = mamba_h;
    return policyNet.forward(state, true);
}

void MPG::reinforce(std::vector<Step>& x, float learningRate)
{
    /*
     * Standard REINFORCE with baseline:
     *   ∇J = E[∇log π(a|s) · (G_t - b)]
     *
     * For softmax policy, gradient w.r.t. logits z:
     *   ∂J/∂z = (G_t - b) · (e_a - π(·|s))
     *
     * We set dLoss[k] = -advantage / π_k so that J·dLoss = -∇J,
     * and RMSProp(θ -= η·g) gives gradient ascent.
     */

    /* Save Mamba state after trajectory for next inference */
    mamba_h = mamba->h;

    /* Compute discounted returns */
    float r = 0;
    Tensor discountedReward(x.size(), 1);
    for (int i = (int)x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();

    /* Reset Mamba for forward pass through the trajectory (training mode) */
    mamba->reset();
    for (std::size_t t = 0; t < x.size(); t++) {
        const Tensor &prob = x[t].action;
        int k = x[t].action.argmax();
        float H = RL::entropy(prob[k]);
        alpha.g[k] += H0 - H;
        x[t].action[k] = prob[k]*(discountedReward[t] - u);
        Tensor &out = policyNet.forward(x[t].state, false);
        Tensor dLoss = Loss::CrossEntropy::df(out, x[t].action);
        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-7, 0.9, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
#endif
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    /* Restore Mamba state for next inference (preserves temporal context) */
    mamba->h = mamba_h;
    return;
}


} // namespace RL
