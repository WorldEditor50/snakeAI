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

/*
   NOTE: none of the three action selectors below restore mamba->h from the
   mamba_h snapshot any more. They used to do `mamba->h = mamba_h;` on entry,
   which clobbered the live recurrent state with the snapshot taken after the
   last training pass before EVERY forward call. Two consequences:
     * inside one rollout the Mamba state could not propagate from one step to
       the next, so the sampled actions carried no temporal information at all;
     * reinforce()/reinforce1() replay the trajectory from a reset state, so the
       distribution they differentiate was not the distribution the actions were
       sampled from — the gradient was biased.
   mamba_h is still saved after each trajectory and restored at the END of the
   replays (see reinforce()/reinforce1()), which is what inference needs.
*/
Tensor &MPG::eGreedyAction(const Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &MPG::noiseAction(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return noise(out);
}

RL::Tensor &MPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = policyNet.forward(state, true);
    return gumbelSoftmax(out, alpha.val);
}

Tensor &MPG::action(const Tensor &state)
{
    /*
       NOTE: mamba->h is deliberately NOT restored from mamba_h here. Restoring
       it on every call reset the recurrence to the state saved before the last
       training pass, so the state could never propagate between two consecutive
       action() calls and the Mamba contributed nothing — a 2-step temporal task
       then degenerates to "map s1 to a fixed action" and scores the 50% chance
       baseline exactly.
       No restore is needed: reinforce()/reinforce1() end with the live state
       equal to the post-trajectory state, which is precisely what mamba_h holds.
    */
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
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    /* Restore Mamba state for next inference (preserves temporal context) */
    mamba->h = mamba_h;
    return;
}

void MPG::reinforce1(std::vector<Step>& x, float learningRate)
{
    /*
        Exact REINFORCE with a mean baseline — the side-effect-free counterpart
        of reinforce() above.

        ∇J = E[ ∇log π(a|s) · (G_t − b) ]
        ∇_z J = A · (e_a − π)                     (softmax policy)

        The softmax layer's backward applies J·dLoss with J_ij = π_i(δ_ij − π_j)
        and dLoss[a] = −A/π_a (0 elsewhere) yields J·dLoss = −∇_z J, so RMSProp's
        θ −= η·g becomes θ += η·∇_z J (ascent).

        Difference from reinforce(): the stored action is only READ.
        reinforce() rewrites x[t].action in place, which scales the gradient by
        the stored probability and is therefore exact only for one-hot stored
        actions — whereas gumbelMax() stores a full distribution.
    */
    const std::size_t n = x.size();
    if (n == 0) {
        return;
    }

    /* State reached after the trajectory: restored at the end so inference can
       keep its temporal context after the training replay. */
    mamba_h = mamba->h;

    /* Discounted returns G_t */
    Tensor discountedReward(n, 1);
    float r = 0;
    for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }

    /* Mean baseline plus a rescale by the return's standard deviation. As in
       DPG::reinforce1, Net::RMSProp's clipGrad normalization divides this global
       factor out again, so it does not change the update here. */
    Tensor advantage(n, 1);
    float u = discountedReward.mean();
    float sd = std::sqrt(discountedReward.variance(u));
    float scale = (sd > 1.0f) ? sd : 1.0f;
    for (std::size_t t = 0; t < n; t++) {
        advantage[t] = (discountedReward[t] - u)/scale;
    }

    /* Replay the trajectory from a clean state so BPTT matches it exactly */
    mamba->reset();
    for (std::size_t t = 0; t < n; t++) {
        int k = x[t].action.argmax();
        Tensor &out = policyNet.forward(x[t].state, false);

        /* Temperature gradient: dJ/dα = H(π) − H_target (see ConvPG::reinforce1) */
        float H = 0;
        for (std::size_t i = 0; i < actionDim; i++) {
            H += RL::entropy(out[i]);
        }
        alpha.g[k] += H - H0;

        float probK = out[k] < 1e-6f ? 1e-6f : out[k];
        Tensor dLoss(actionDim, 1);      /* zero-filled */
        dLoss[k] = -advantage[t]/probK;  /* negative = gradient ascent */

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
    alpha.clamp(0.2f, 0.2f, 1.0f);
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    mamba->h = mamba_h;
    return;
}


} // namespace RL
