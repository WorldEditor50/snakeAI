#include "drpg.h"
#include "layer.h"
#include "loss.h"

RL::DRPG::DRPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.9), exploringRate(1)
{
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    H0 = RL::entropy(0.25);
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
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::DRPG::action(const Tensor &state)
{
    /*
       NOTE: the LSTM state is deliberately NOT restored from the h/c members
       here. Doing so made every call reset the recurrence to the state saved
       before the last training pass, so the state could never propagate between
       two consecutive action() calls and the recurrent pathway carried no
       information at all — a 2-step temporal task then degenerates to "map s1 to
       a fixed action" and scores exactly the 50% chance baseline.
       No restore is needed: reinforce()/reinforce1() leave the live state equal
       to the post-trajectory state (they replay the same trajectory from reset
       with unchanged weights), which is exactly what h/c hold.
    */
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
    return;
}

void RL::DRPG::reinforce1(std::vector<Step>& x, float learningRate)
{
    /*
        Standard REINFORCE with baseline:
            ∇J = E[∇log π(a|s) · (G_t - b)]

        For softmax policy, the gradient w.r.t. logits z is:
            ∂J/∂z = (G_t - b) · (e_a - π(·|s))

        This is achieved by setting dLoss[i] such that
        J(softmax) · dLoss = advantage · (e_k - p):
            dLoss[k] = advantage / p[k],  dLoss[i≠k] = 0

        where J is the softmax Jacobian, e_k is one-hot at the selected action.
    */
    const std::size_t n = x.size();
    if (n == 0) {
        return;
    }

    /* LSTM state reached after the trajectory: kept so inference continues from
       the same point once training has replayed the sequence. */
    h = lstm->h;
    c = lstm->c;

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

    /* Replay the trajectory from a clean LSTM state so BPTT matches it exactly */
    lstm->reset();
    for (std::size_t t = 0; t < n; t++) {
        int k = x[t].action.argmax();

        /* --- Forward pass to get current policy distribution --- */
        Tensor &out = policyNet.forward(x[t].state, false);

        /* --- alpha (temperature) gradient ---
           Entropy of the FULL policy distribution. The SAC dual objective is
           J(α) = α·(H(π) − H_target) so dJ/dα = H(π) − H_target; RMSProp then
           lowers α when the policy is too random. The previous version also
           multiplied by alpha[k], which is not part of the derivation. */
        float policyEntropy = 0;
        for (std::size_t i = 0; i < actionDim; i++) {
            policyEntropy += RL::entropy(out[i]);
        }
        alpha.g[k] += policyEntropy - H0;

        /* --- Standard REINFORCE policy gradient (ascent via negated dLoss) --- */
        float probK = out[k] < 1e-6f ? 1e-6f : out[k];   /* guard 1/π blow-up */
        Tensor dLoss(actionDim, 1);
        dLoss.zero();
        dLoss[k] = -advantage[t]/probK;

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
    alpha.clamp(0.2f, 0.2f, 1.0f);
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    return;
}
