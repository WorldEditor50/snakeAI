#include "bcq.h"
#include "layer.h"
#include "loss.h"

RL::BCQ::BCQ(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), learningSteps(0)
{
    featureDim = stateDim + actionDim;
    /*
     * BCQ VAE: models P(s,a) via reconstruction.
     *   input: concat(state, action)  — dim = featureDim = 8
     *   latent z: stateDim = 4
     *   output: [s_reconstructed, a_candidate] — dim = 8, Tanh → [-1, 1]
     */
    encoder = VAE(featureDim, 2*featureDim, stateDim);
    /*
     * Actor: maps concat(state, VAE_candidate_action) → action probabilities
     *   input:  state(4) + candidate_action(4) = 8
     *   output: actionDim (4) = softmax probabilities
     */
    actor = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));
    /*
     * Critics: Clipped Double Q-learning (Linear/Unbounded output)
     *   input:  concat(state, action) = 8
     *   output: Q(s,a) for each action dim = 4
     *
     *   FIX: Linear output (was Sigmoid) to support TD-target ∈ [-1.5, 2.49]
     */
    for (int i = 0; i < max_qnet_num; i++) {
        critics[i] = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, true),
                         TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                         Layer<Linear>::_(hiddenDim, actionDim, true, true));
        criticsTarget[i] = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, false),
                               TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                               Layer<Linear>::_(hiddenDim, actionDim, true, false));
        critics[i].copyTo(criticsTarget[i]);
    }

    /* allocate output buffer */
    actionOut = Tensor(actionDim, 1);
}

RL::Tensor &RL::BCQ::action(const Tensor &state)
{
    /*
     * Standard BCQ discrete action selection:
     *   1) Sample NUM_CANDIDATES times from VAE prior z ~ N(0, I)
     *   2) Decode each to get candidate (s_hat, a_hat)
     *   3) Evaluate Q(s, a_hat) using Clipped Double Q
     *   4) Pick action with highest Q value
     */
    float bestQ = -1e10;
    actionOut.zero();

    for (int c = 0; c < NUM_CANDIDATES; c++) {
        /* Sample from prior: z ~ N(0,I), decode → 8-dim = [s_hat, a_hat] */
        const Tensor& decoded = encoder.priorDecode();
        /* Extract action part: last actionDim elements */
        Tensor candidateAction = decoded.block({stateDim, 0}, {actionDim, 1});
        /* Concatenate with TRUE state for critic evaluation */
        Tensor sa = Tensor::concat(1, state, candidateAction);

        /* Clipped Double Q: min(Q1, Q2) */
        float qCandidate = 1e10;
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = critics[i].forward(sa);
            /* Evaluate at the action this candidate suggests */
            std::size_t bestA = candidateAction.argmax();
            qCandidate = std::min(qCandidate, qi[bestA]);
        }

        if (qCandidate > bestQ) {
            bestQ = qCandidate;
            actionOut.zero();
            actionOut[candidateAction.argmax()] = 1;
        }
    }

    return actionOut;
}

RL::Tensor &RL::BCQ::mixAction(const RL::Tensor &state, const RL::Tensor &ga)
{
    Tensor sa = Tensor::concat(1, state, ga);
    Tensor& a = actor.forward(sa);
    a += ga;
    return a;
}


void RL::BCQ::perceive(const RL::Tensor &state,
                       const RL::Tensor &action,
                       const RL::Tensor &nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void RL::BCQ::experienceReplay(const Transition& x)
{
    /* ============ 1. train encoder (VAE reconstruction of (s,a) pairs) ============ */
    {
        Tensor sa = Tensor::concat(1, x.state, x.action);  // 8-dim
        encoder.forward(sa);
        encoder.backward(sa);
    }

    /* ============ 2. train critics (TD-learning with Clipped Double Q) ============ */
    {
        /* Build true (s, a) input for current Q evaluation */
        Tensor sa_taken = Tensor::concat(1, x.state, x.action);   // 8-dim
        std::size_t actionTaken = x.action.argmax();

        /*
         * TD-target using Clipped Double Q + VAE candidate generation:
         *
         *   y = r + γ · max_{a_cand} min(Q₁'(s', a_cand), Q₂'(s', a_cand))
         *
         * where a_cand comes from VAE prior sampling z ~ N(0,I)
         */
        float bestMaxQ = -1e10;
        for (int c = 0; c < NUM_CANDIDATES; c++) {
            /* Sample from prior: z ~ N(0,I) */
            const Tensor& decoded = encoder.priorDecode();    // 8-dim
            Tensor candidateAction = decoded.block({stateDim, 0}, {actionDim, 1});

            /* Evaluate on TRUE next state (not VAE reconstructed) */
            Tensor sa_next = Tensor::concat(1, x.nextState, candidateAction);

            /* Clipped Double Q: min over all target critics */
            float qCandidate = 1e10;
            for (int i = 0; i < max_qnet_num; i++) {
                const Tensor& qi = criticsTarget[i].forward(sa_next);
                std::size_t bestA = candidateAction.argmax();
                qCandidate = std::min(qCandidate, qi[bestA]);
            }

            if (qCandidate > bestMaxQ) {
                bestMaxQ = qCandidate;
            }
        }

        /* TD target */
        float tdTarget = x.done ? x.reward : x.reward + gamma * bestMaxQ;

        /* Update each critic with MSE loss on the taken action */
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor &out = critics[i].forward(sa_taken);
            Tensor qTarget = out;
            qTarget[actionTaken] = tdTarget;
            critics[i].backward(sa_taken, Loss::MSE::df(out, qTarget));
        }
    }

    /* ============ 3. train actor (policy gradient: maximize Q) ============ */
    {
        /*
         * Sample candidate from VAE prior, use actor to refine action selection.
         * Actor input:  concat(state, candidate_action) = 8-dim
         * Actor output: refined action probabilities = 4-dim (softmax)
         * Policy gradient: ∇J = Σ Q(s,a) · ∇π(a|s, a_cand)
         */
        const Tensor& decoded = encoder.priorDecode();          // 8-dim
        Tensor candidateAction = decoded.block({stateDim, 0}, {actionDim, 1});
        Tensor sa = Tensor::concat(1, x.state, candidateAction);

        actor.forward(sa);                                       // 8-dim → 4-dim

        /* Compute Q-values using Clipped Double Q */
        Tensor q(actionDim, 1);
        for (int a = 0; a < actionDim; a++) {
            Tensor oneHot(actionDim, 1);
            oneHot.zero();
            oneHot[a] = 1;
            Tensor sa_a = Tensor::concat(1, x.state, oneHot);

            /* Clipped Double Q */
            float qVal = 1e10;
            for (int i = 0; i < max_qnet_num; i++) {
                const Tensor& qi = critics[i].forward(sa_a);
                qVal = std::min(qVal, qi[a]);
            }
            q[a] = qVal;
        }

        /* Policy gradient: maximize Σ π(a) · Q(s,a) */
        actor.backward(sa, q);
    }
    return;
}


void RL::BCQ::learn(std::size_t maxMemorySize,
                    std::size_t /*replaceTargetIter*/,
                    std::size_t batchSize,
                    float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    learningSteps++;

    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* Polyak soft-update target networks */
    for (int i = 0; i < max_qnet_num; i++) {
        critics[i].softUpdateTo(criticsTarget[i], 5e-3);
    }

    /* Adam optimizer — reduced learning rate for offline RL stability */
    float lr = 3e-4;  // was 1e-3, lowered
    actor.Adam(lr, 0.99, 0.9, 1e-4);
    encoder.Adam(lr, 0.99, 0.9, 1e-4);
    for (int i = 0; i < max_qnet_num; i++) {
        critics[i].Adam(lr, 0.99, 0.9, 1e-4);
    }

    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    return;
}
