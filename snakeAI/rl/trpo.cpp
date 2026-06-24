#include "trpo.h"
#include "layer.h"
#include "loss.h"

/* number of layers in actor network (3 layers in constructor) */
static const int ACTOR_LAYERS = 3;

RL::TRPO::TRPO(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99)
{
    /* Per-sample KL bound. Balanced for GAE advantages. */
    maxKL = 0.002;
    learningSteps = 0;
    exploringRate = 1;

    actorP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    actorQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, false));

    /* Copy initial weights from actorP to actorQ */
    actorP.copyTo(actorQ);

    /* V(s) state-value critic: state -> hidden -> hidden -> 1 scalar */
    critic = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Linear>::_(hiddenDim, 1, true, true));
}

RL::Tensor &RL::TRPO::eGreedyAction(const RL::Tensor &state)
{
    Tensor& out = actorQ.forward(state);
    return eGreedy(out, exploringRate, true);
}

RL::Tensor &RL::TRPO::action(const RL::Tensor &state)
{
    return actorP.forward(state);
}

/* ----- TRPO helper: flatten/unflatten parameters ----- */

int RL::TRPO::totalParams()
{
    int total = 0;
    for (int i = 0; i < ACTOR_LAYERS; i++) {
        iFcLayer *layer = static_cast<iFcLayer*>(actorP[i]);
        total += (int)layer->w.totalSize;
        if (layer->bias) {
            total += (int)layer->b.totalSize;
        }
    }
    return total;
}

RL::Tensor RL::TRPO::flatParams()
{
    int n = totalParams();
    Tensor params(n, 1);
    int idx = 0;
    for (int i = 0; i < ACTOR_LAYERS; i++) {
        iFcLayer *layer = static_cast<iFcLayer*>(actorP[i]);
        for (std::size_t j = 0; j < layer->w.totalSize; j++) {
            params[idx++] = layer->w[j];
        }
        if (layer->bias) {
            for (std::size_t j = 0; j < layer->b.totalSize; j++) {
                params[idx++] = layer->b[j];
            }
        }
    }
    return params;
}

void RL::TRPO::setFlatParams(const Tensor &p)
{
    int idx = 0;
    for (int i = 0; i < ACTOR_LAYERS; i++) {
        iFcLayer *layer = static_cast<iFcLayer*>(actorP[i]);
        for (std::size_t j = 0; j < layer->w.totalSize; j++) {
            layer->w[j] = p[idx++];
        }
        if (layer->bias) {
            for (std::size_t j = 0; j < layer->b.totalSize; j++) {
                layer->b[j] = p[idx++];
            }
        }
    }
}

RL::Tensor RL::TRPO::flatGrad()
{
    int n = totalParams();
    Tensor g(n, 1);
    int idx = 0;
    for (int i = 0; i < ACTOR_LAYERS; i++) {
        iFcLayer *layer = static_cast<iFcLayer*>(actorP[i]);
        for (std::size_t j = 0; j < layer->g.w.totalSize; j++) {
            g[idx++] = layer->g.w[j];
        }
        if (layer->bias) {
            for (std::size_t j = 0; j < layer->g.b.totalSize; j++) {
                g[idx++] = layer->g.b[j];
            }
        }
    }
    return g;
}

void RL::TRPO::zeroGrad()
{
    for (int i = 0; i < ACTOR_LAYERS; i++) {
        iFcLayer *layer = static_cast<iFcLayer*>(actorP[i]);
        layer->g.zero();
    }
}

/*
 * Policy gradient: dLoss[k] = -adv / π[k]
 * After softmax backward (J^T), the result is adv·(e_k - π).
 */
/*
 * Policy gradient: dLoss[k] = -adv / π_k
 * Softmax backward J^T maps this to -A·(e_k - π) = ∇_z (-J).
 * So g = w.grad = -∇_θ J (negative policy gradient).
 * CG: H·x_cg = g = -∇_θ J → x_cg = -H^{-1}·∇_θ J.
 * Negate x_cg for ascent: Δθ = -x_cg = H^{-1}·∇_θ J.
 * RMSProp/manual: w -= lr · g = w += lr · ∇_θ J (ascent).
 */
static RL::Tensor makePolicyGradLoss(int actionDim, int k, float adv, float probK)
{
    RL::Tensor dLoss(actionDim, 1);
    dLoss.zero();
    dLoss[k] = -adv / (probK + 1e-9f);
    return dLoss;
}

static void accumulatePolicyGrad(RL::Net &net,
                                  std::size_t N,
                                  const std::vector<RL::Tensor> &states,
                                  const std::vector<int> &actionIndices,
                                  const RL::Tensor &advantages,
                                  int actionDim)
{
    for (std::size_t i = 0; i < N; i++) {
        int k = actionIndices[i];
        float adv = advantages[i];
        RL::Tensor &p = net.forward(states[i]);
        float pk = p[k];
        RL::Tensor dLoss = makePolicyGradLoss(actionDim, k, adv, pk);
        net.backward(states[i], dLoss);
    }
}

static void accumulateKLGrad(RL::Net &net,
                              std::size_t N,
                              const std::vector<RL::Tensor> &states,
                              const std::vector<RL::Tensor> &oldProbs,
                              int actionDim)
{
    for (std::size_t i = 0; i < N; i++) {
        RL::Tensor &p = net.forward(states[i]);
        RL::Tensor dKL(actionDim, 1);
        for (int j = 0; j < actionDim; j++) {
            /* Clamp denominator to prevent division by near-zero
               which would produce extreme gradients and NaN. */
            float pj = p[j] < 1e-6f ? 1e-6f : p[j];
            dKL[j] = -oldProbs[i][j] / pj;
        }
        net.backward(states[i], dKL);
    }
}

static float computeAvgKL(RL::Net &net,
                           std::size_t N,
                           const std::vector<RL::Tensor> &states,
                           const std::vector<RL::Tensor> &oldProbs,
                           int actionDim)
{
    float kl = 0;
    for (std::size_t i = 0; i < N; i++) {
        RL::Tensor &p = net.forward(states[i]);
        for (int j = 0; j < actionDim; j++) {
            if (oldProbs[i][j] > 1e-9f && p[j] > 1e-9f) {
                /* Clamp ratio denominator to avoid log(inf) → NaN */
                float ratio = oldProbs[i][j] / (p[j] < 1e-9f ? 1e-9f : p[j]);
                kl += oldProbs[i][j] * std::log(ratio);
            }
        }
    }
    return kl / float(N);
}

static float computeAvgSurrogate(RL::Net &net,
                                  std::size_t N,
                                  const std::vector<RL::Tensor> &states,
                                  const std::vector<int> &actionIndices,
                                  const std::vector<RL::Tensor> &oldProbs,
                                  const RL::Tensor &advantages)
{
    float surr = 0;
    for (std::size_t i = 0; i < N; i++) {
        RL::Tensor &p = net.forward(states[i]);
        int k = actionIndices[i];
        float ratio = p[k] / (oldProbs[i][k] + 1e-9f);
        surr += ratio * advantages[i];
    }
    return surr / float(N);
}

void RL::TRPO::learn(std::vector<RL::Step> &x, float learningRate)
{
    std::size_t N = x.size();
    if (N < 1) {
        return;
    }

    (void)learningRate; /* TRPO does not use a fixed LR; step size comes from KL constraint */

    const float gaeLambda = 0.95f;

    /* === Compute MC returns for critic training === */
    std::vector<float> mcReturns(N, 0.0f);
    {
        int end = (int)N - 1;
        mcReturns[end] = x[end].reward;
        for (int i = end - 1; i >= 0; i--) {
            mcReturns[i] = x[i].reward + gamma * mcReturns[i + 1];
        }
    }

    /* === Train critic V(s) on MC returns === */
    {
        for (std::size_t i = 0; i < N; i++) {
            Tensor &v = critic.forward(x[i].state);
            Tensor target(1, 1);
            target[0] = mcReturns[i];
            critic.backward(x[i].state, Loss::MSE::df(v, target));
        }
        critic.RMSProp(1e-3f, 0.9f, 0.0f);
    }

    /* === Compute GAE advantages ===
       δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
       A_t = Σ_{l=0} (γλ)^l · δ_{t+l}
       GAE generalizes TD(λ=0) and MC(λ=1). For independent bandit steps,
       V(s_{t+1}) acts as a baseline, giving correct per-step advantages. */
    Tensor advantages(N, 1);
    {
        /* Compute TD errors */
        std::vector<float> tdErrors(N, 0.0f);
        for (std::size_t i = 0; i < N; i++) {
            Tensor &v = critic.forward(x[i].state);
            float v_curr = v[0];
            float v_next = 0.0f;
            if (i + 1 < N) {
                Tensor &vn = critic.forward(x[i + 1].state);
                v_next = vn[0];
            }
            tdErrors[i] = x[i].reward + gamma * v_next - v_curr;
        }

        /* Backward GAE accumulation */
        float gae = 0.0f;
        for (int i = (int)N - 1; i >= 0; i--) {
            gae = tdErrors[i] + gamma * gaeLambda * gae;
            advantages[i] = gae;
        }
    }

    /* === Normalize advantages (reduce variance) ===
     * Only normalize when N > 1, otherwise single-step
     * advantages are zeroed by mean-subtraction.
     */
    if (N > 1) {
        float advMean = advantages.mean();
        float advStd = advantages.variance(advMean);
        advStd = std::sqrt(advStd + 1e-9f);
        if (advStd > 1e-9f) {
            for (std::size_t i = 0; i < N; i++) {
                advantages[i] = (advantages[i] - advMean) / advStd;
            }
        }
    }


    /* Extract states, store old action probs and action indices */
    std::vector<Tensor> states(N);
    std::vector<Tensor> oldProbs(N);
    std::vector<int> actionIndices(N);
    for (std::size_t i = 0; i < N; i++) {
        states[i] = x[i].state;
        Tensor &oldProb = actorP.forward(x[i].state);
        oldProbs[i] = oldProb;  // copy
        actionIndices[i] = (int)x[i].action.argmax();
    }

    /* === Compute policy gradient g = ∇_θ L(θ) === */
    zeroGrad();
    accumulatePolicyGrad(actorP, N, states, actionIndices,
                         advantages, actionDim);
    Tensor policyGrad = flatGrad();
    zeroGrad();

    /* Check gradient norm */
    float gradNorm = std::sqrt(Tensor::dot(policyGrad, policyGrad) + 1e-12f);
    if (gradNorm < 1e-12f) {
        return;
    }

    const int cgIter = 15;
    const float damping = 0.1f;

    /*
     * === Conjugate Gradient: solve Hx = g ===
     */
    int nParams = totalParams();
    Tensor x_cg(nParams, 1);
    x_cg.zero();
    {
        Tensor r = policyGrad;
        Tensor p_dir = r;
        float rsold = Tensor::dot(r, r);

        for (int iter = 0; iter < cgIter; iter++) {

            if (rsold < 1e-12f) break;

            float pNorm = std::sqrt(Tensor::dot(p_dir, p_dir)) + 1e-12f;
            float eps = 1e-5f / pNorm;

            Tensor params = flatParams();

            /* g_plus = ∇KL(θ + ε·p_dir) */
            setFlatParams(params + p_dir * eps);
            zeroGrad();
            accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
            Tensor g_plus = flatGrad();
            zeroGrad();

            /* g_minus = ∇KL(θ - ε·p_dir) */
            setFlatParams(params - p_dir * eps);
            zeroGrad();
            accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
            Tensor g_minus = flatGrad();
            zeroGrad();

            setFlatParams(params);

            Tensor Hp = (g_plus - g_minus) / (2.0f * eps);
            for (int j = 0; j < nParams; j++) {
                Hp[j] += damping * p_dir[j];
            }

            float pHp = Tensor::dot(p_dir, Hp);
            if (pHp <= 0) {
                p_dir = r;
                pNorm = std::sqrt(Tensor::dot(p_dir, p_dir)) + 1e-12f;
                eps = 1e-5f / pNorm;

                setFlatParams(params + p_dir * eps);
                zeroGrad();
                accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
                g_plus = flatGrad();
                zeroGrad();

                setFlatParams(params - p_dir * eps);
                zeroGrad();
                accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
                g_minus = flatGrad();
                zeroGrad();

                setFlatParams(params);

                Hp = (g_plus - g_minus) / (2.0f * eps);
                for (int j = 0; j < nParams; j++) {
                    Hp[j] += damping * p_dir[j];
                }
                pHp = Tensor::dot(p_dir, Hp);
                if (pHp <= 0) break;
            }

            float alpha_cg = rsold / pHp;
            x_cg += p_dir * alpha_cg;
            r -= Hp * alpha_cg;

            float rsnew = Tensor::dot(r, r);
            if (rsnew < 1e-12f) break;

            float beta = rsnew / rsold;
            p_dir = r + p_dir * beta;
            rsold = rsnew;
        }
    }


    float dirNorm = std::sqrt(Tensor::dot(x_cg, x_cg) + 1e-12f);
    if (dirNorm < 1e-12f) {
        /* Fallback: vanilla policy gradient
         * accumulatePolicyGrad fills g = -∇_θ J (negative PG).
         * RMSProp does w -= lr·g = w += lr·∇_θ J (ascent). OK.
         */
        zeroGrad();
        accumulatePolicyGrad(actorP, N, states, actionIndices,
                             advantages, actionDim);
        actorP.RMSProp(1e-3f, 0.9f, 0.0f);
        learningSteps++;
        actorP.copyTo(actorQ);
        exploringRate *= 0.999f;
        exploringRate = exploringRate < 0.1f ? 0.1f : exploringRate;
        return;
    }



    /*
     * === Compute step size ===
     * s = sqrt(2 * maxKL / (x_cg^T · H · x_cg))
     */
    float stepSize;
    {
        float dirNorm2 = std::sqrt(Tensor::dot(x_cg, x_cg)) + 1e-12f;
        float eps = 1e-5f / dirNorm2;

        Tensor params = flatParams();

        setFlatParams(params + x_cg * eps);
        zeroGrad();
        accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
        Tensor gp = flatGrad();
        zeroGrad();

        setFlatParams(params - x_cg * eps);
        zeroGrad();
        accumulateKLGrad(actorP, N, states, oldProbs, actionDim);
        Tensor gm = flatGrad();
        zeroGrad();

        setFlatParams(params);

        Tensor Hdir = (gp - gm) / (2.0f * eps);
        for (int j = 0; j < nParams; j++) {
            Hdir[j] += damping * x_cg[j];
        }
        float dir_H_dir = Tensor::dot(x_cg, Hdir);
        if (dir_H_dir <= 0) {
            dir_H_dir = Tensor::dot(x_cg, x_cg) + 1e-12f;
        }
        /* N correction: dir_H_dir = N * (x^T * FIM_avg * x).
           Step size for per-sample constraint: sqrt(2*maxKL*N/dir_H_dir) */
        stepSize = std::sqrt(2.0f * maxKL * float(N) / dir_H_dir);
    }

    /* CG solves H·x_cg = g where g = w.grad = -∇_θ J (negative policy gradient).
     * x_cg = -H^{-1}·∇_θ J (natural gradient descent direction).
     * Negate for ascent: stepDir = -x_cg = H^{-1}·∇_θ J.
     */
    Tensor stepDir = -x_cg;



    /* === Backtracking Line Search === */

    Tensor oldParams = flatParams();
    bool found = false;
    float bestAlpha = 0.0f;
    float bestKLval = 1e10f;

    /* Old surrogate: at θ=θ_old, ratio=1, so oldSurr = mean(advantages) */
    float oldSurr = 0;
    for (std::size_t i = 0; i < N; i++) {
        oldSurr += advantages[i];
    }
    oldSurr /= float(N);

    const int maxBacktrack = 10;

    for (int bt = 0; bt < maxBacktrack; bt++) {
        float alpha_ls = std::pow(0.5f, bt);
        Tensor newParams = oldParams + stepDir * (stepSize * alpha_ls);
        setFlatParams(newParams);

        float kl = computeAvgKL(actorP, N, states, oldProbs, actionDim);
        float newSurr = computeAvgSurrogate(actorP, N, states, actionIndices,
                                             oldProbs, advantages);

        if (std::isnan(kl) || std::isinf(kl) ||
            std::isnan(newSurr) || std::isinf(newSurr)) {
            continue;
        }

        /* Accept if KL constraint and surrogate improvement are satisfied */
        if (kl <= maxKL * 1.5f && newSurr >= oldSurr - 1e-6f) {
            bestAlpha = alpha_ls;
            found = true;
            break;
        }

        if (kl < bestKLval) {
            bestKLval = kl;
            bestAlpha = alpha_ls;
        }
    }

    if (!found) {
        if (bestAlpha > 0 && bestKLval < maxKL * 5.0f) {
            /* Accept best candidate */
            Tensor newParams = oldParams + stepDir * (stepSize * bestAlpha);
            setFlatParams(newParams);
        } else {
            /* Fallback: vanilla policy gradient */
            setFlatParams(oldParams);
            zeroGrad();
            accumulatePolicyGrad(actorP, N, states, actionIndices,
                                 advantages, actionDim);
            actorP.RMSProp(1e-3f, 0.9f, 0.0f);
        }
    }

    learningSteps++;

    /* Sync exploration network with trained policy */
    actorP.copyTo(actorQ);

    /* Decay exploration rate */
    exploringRate *= 0.999f;
    exploringRate = exploringRate < 0.1f ? 0.1f : exploringRate;
}

void RL::TRPO::save(const std::string &actorPara, const std::string &criticPara)
{
    actorP.save(actorPara);
    critic.save(criticPara);
}

void RL::TRPO::load(const std::string &actorPara, const std::string &criticPara)
{
    actorP.load(actorPara);
    actorP.copyTo(actorQ);
    critic.load(criticPara);
}
