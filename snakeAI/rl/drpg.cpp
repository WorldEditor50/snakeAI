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
        const Tensor &oneHotAction = x[t].action;
        int k = oneHotAction.argmax();

        /* --- Forward pass to get current policy distribution --- */
        Tensor &out = policyNet.forward(x[t].state, false);

        /* --- alpha (temperature) gradient ---
           Entropy computed from FULL policy distribution, not one-hot action */
        float policyEntropy = 0;
        for (std::size_t i = 0; i < actionDim; i++) {
            policyEntropy += RL::entropy(out[i]);
        }
        alpha.g[k] += (policyEntropy - H0) * alpha[k];

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
        float advantage = discountedReward[t] - u;
        Tensor dLoss(actionDim, 1);
        dLoss.zero();
        dLoss[k] = -advantage / (out[k] + 1e-9f);

        policyNet.backward(x[t].state, dLoss);
    }
    alpha.RMSProp(1e-5, 0.9, 0);
    policyNet.RMSProp(learningRate, 0.9, 0);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.25 ? 0.25 : exploringRate;
    return;
}
