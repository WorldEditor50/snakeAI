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
    /*
       Conv2d::_ argument order is
           (inChannels, h, w, outChannels, kernelSize, stride, padding, bias, withGrad)
       so the first layer is k=5, stride=5, padding=1 and maps
           1x118x118 -> 4x24x24          ((118-5+2)/5 + 1 = 24)
       then MaxPooling2d(k=2,s=2) -> 4x12x12
       then Conv2d k=3, stride=3  -> 8x4x4   ((12-3)/3 + 1 = 4)
       then MaxPooling2d(k=2,s=2) -> 8x2x2
       then Layer<Tanh>(8*2*2 -> hiddenDim).
       These declarations are internally consistent: every layer's declared h/w
       equals the previous layer's real output, so the whole 118x118 board is
       covered (with stride 5) rather than cropped.
    */
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

void RL::ConvPG::reinforce1(std::vector<Step>& x, float learningRate)
{
    /*
        Exact REINFORCE with a mean baseline — the side-effect-free counterpart
        of reinforce() above.

        ∇J = E[ ∇log π(a|s) · (G_t − b) ]
        ∇_z J = A · (e_a − π)                    (softmax policy)

        The softmax layer's backward applies J·dLoss with J_ij = π_i(δ_ij − π_j).
        With dLoss[a] = −A/π_a and 0 elsewhere that gives
            J·dLoss = −A·(e_a − π) = −∇_z J,
        so RMSProp's θ −= η·g becomes θ += η·∇_z J (ascent).

        Difference from reinforce(): the stored action is only READ.
        reinforce() rewrites x[t].action in place, which scales the gradient by
        the stored probability and is therefore exact only when that action is a
        one-hot vector — whereas gumbelMax() stores a full distribution.
    */
    const std::size_t n = x.size();
    if (n == 0) {
        return;
    }

    /* Discounted returns G_t */
    Tensor discountedReward(n, 1);
    float r = 0;
    for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }

    /* Mean baseline plus a rescale by the return's standard deviation (divisor
       floored at 1 so a nearly-flat return sequence cannot have its noise
       amplified). As in DPG::reinforce1, this global factor is divided out again
       by Net::RMSProp's clipGrad normalization, so it does not change the update
       here; it only matters under an optimizer called with clipGrad = false. */
    Tensor advantage(n, 1);
    float u = discountedReward.mean();
    float sd = std::sqrt(discountedReward.variance(u));
    float scale = (sd > 1.0f) ? sd : 1.0f;
    for (std::size_t t = 0; t < n; t++) {
        advantage[t] = (discountedReward[t] - u)/scale;
    }

    for (std::size_t t = 0; t < n; t++) {
        int k = x[t].action.argmax();
        Tensor &out = policyNet.forward(x[t].state);

        /* Temperature (alpha) gradient. The SAC dual objective is
           J(α) = α·(H(π) − H_target), so dJ/dα = H(π) − H_target; RMSProp then
           applies α −= η·g, which lowers α when the policy is too random.
           (Note: DPG's method used the opposite sign.) */
        float H = 0;
        for (std::size_t i = 0; i < actionDim; i++) {
            H += RL::entropy(out[i]);
        }
        alpha.g[k] += H - entropy0;

        float probK = out[k] < 1e-6f ? 1e-6f : out[k];
        Tensor dLoss(actionDim, 1);      /* zero-filled */
        dLoss[k] = -advantage[t]/probK;  /* negative = gradient ascent */

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
    alpha.clamp(0.2f, 0.2f, 1.0f);
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

