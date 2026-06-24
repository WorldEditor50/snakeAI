#include "sac.h"
#include "layer.h"
#include "loss.h"
#include <limits>
#include "moe.hpp"

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1), learningSteps(0)
{
    annealing = ExpAnnealing(0.25, 1, 1e-7);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(0.65);
    /* target entropy = -actionDim (standard SAC heuristic) */
    H0 = -std::log(actionDim);
    actor = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    for (int i = 0; i < QNET_NUM; i++) {
        /* Sigmoid output keeps Q in (0,1) — symmetric gradient around
           the midpoint of reward targets, preventing saturation asymmetry. */
        critics[i] = Net(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true, true),
                         TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                         Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

        criticsTarget[i] = Net(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true, false),
                               TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                               Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
        /* Independent random init + training naturally breaks symmetry */
        critics[i].copyTo(criticsTarget[i]);
    }
}

void RL::SAC::perceive(const Tensor& state,
                       const Tensor& action,
                       const Tensor& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor &RL::SAC::eGreedyAction(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return eGreedy(out, exploringRate, true);
}

RL::Tensor &RL::SAC::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);
    return RL::gumbelSoftmax(out, 0.9);
}

RL::Tensor& RL::SAC::action(const RL::Tensor &state)
{
    return actor.forward(state);
}

void RL::SAC::experienceReplay(const RL::Transition &x)
{
    /* ================================================
     * 1. Compute Q-target using Clipped Double Q-learning
     *    with expectation over actions (not argmax)
     * ================================================ */
    /* Cache target critic outputs — each is forwarded ONCE */
    const Tensor* targetQs[QNET_NUM];
    /* Get policy probabilities for next state (reused below) */
    const Tensor &nextProb = actor.forward(x.nextState);
    Tensor nextState = Tensor::concat(0, x.nextState, nextProb);
    for (int i = 0; i < QNET_NUM; i++) {
        targetQs[i] = &criticsTarget[i].forward(nextState);
    }
    /* Q-target for the taken action only */
    std::size_t k = x.action.argmax();
    /* Compute min Q over all target critics (element-wise per action) */
    float minQTarget = std::numeric_limits<float>::max();
    for (int i = 0; i < QNET_NUM; i++) {
        minQTarget = std::min(minQTarget, (*targetQs[i])[k]);
    }

    /* Train each critic with MSE loss — forward each online critic ONCE */
    const Tensor &prob = actor.forward(x.state);
    Tensor state = Tensor::concat(0, x.state, prob);
    for (int i = 0; i < QNET_NUM; i++) {
        const Tensor &out = critics[i].forward(state);
        Tensor qTarget = out;
        /* V(s') = Σ π(a'|s') * (minQ(s',a') - α·log(π(a'|s'))) */
        float nextValue = nextProb[k] * (minQTarget - alpha[k]*std::log(nextProb[k] + 1e-8));
        qTarget[k] = x.reward + (1 - x.done)*gamma*nextValue;
        critics[i].backward(state, Loss::MSE::df(out, qTarget));
    }

    /* ================================================
     * 2. Train Policy Net
     *    J(π) = Σ π(a|s) * (α·log(π(a|s)) - minQ(s,a))
     *    gradient w.r.t π output: α·log(π) + α - Q
     * ================================================ */
    /* Cache online critic outputs — each forwarded ONCE */
    const Tensor* onlineQs[QNET_NUM];
    Tensor prob_ = prob;  /* reused in step 3 */
    //gumbelSoftmax(prob_, 0.9);
    Tensor state_ = Tensor::concat(0, x.state, prob_);
    for (int i = 0; i < QNET_NUM; i++) {
        onlineQs[i] = &critics[i].forward(state_);
    }

    /* Compute min Q over all online critics (element-wise) */
    Tensor minQ(actionDim, 1);
    for (int j = 0; j < actionDim; j++) {
        float q_min = std::numeric_limits<float>::max();
        for (int i = 0; i < QNET_NUM; i++) {
            q_min = std::min(q_min, (*onlineQs[i])[j]);
        }
        minQ[j] = q_min;
    }

    /* SAC policy gradient error on softmax output:
     * dJ/dπ(a|s) = α·log(π(a|s)) + α - Q(s,a)
     * The softmax layer's gradient function handles the
     * backpropagation through the softmax nonlinearity. */
    Tensor loss(actionDim, 1);
    for (int i = 0; i < actionDim; i++) {
        loss[i] = -minQ[i] + alpha[i]*(std::log(prob_[i] + 1e-8) + 1);
    }
    actor.backward(x.state, loss);

    /* ================================================
     * 3. Update Temperature (alpha)
     *    J(α) = -α * (H - H₀)  where  H₀ = -|A| = entropy0
     *    ∇α = H₀ - H
     * ================================================ */
    {
        /* Reuse policy p from step 2 — no extra forward pass */
        float  H = 0;
        for (int i = 0; i < actionDim; i++) {
            H += RL::entropy(prob[i]);
        }
        /* ∇α = -(H - H₀) = H₀ - H */
        float alphaGrad = H0 - H;
        alpha.g[k] += alphaGrad;
    }
    return;
}

void RL::SAC::learn(size_t maxMemorySize, size_t replaceTargetIter, size_t batchSize, float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        /* Polyak averaging with consistent tau for all critics */
        float tau = 1e-3;
        for (int i = 0; i < QNET_NUM; i++) {
            critics[i].softUpdateTo(criticsTarget[i], tau);
            tau += 2e-3;
        }
        learningSteps = 0;
    }

    /* Experience replay with random mini-batch */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        experienceReplay(memories[k]);
    }

    /* Apply gradient updates */
    actor.RMSProp(1e-2, 0.9, 0);

#if 1
    std::cout<<"annealing:"<<annealing.val<<",alpha:";
    alpha.val.printValue();
#endif

    alpha.RMSProp(1e-7, 0.9, 1e-6);
    /* Keep alpha in reasonable range */
    alpha.clamp(0.25, 0.64, 1);
    annealing.step();

    for (int i = 0; i < QNET_NUM; i++) {
        critics[i].RMSProp(1e-3, 0.9, 0);
    }

    /* manage replay buffer: drop oldest entries when full */
    if (memories.size() > maxMemorySize) {
        std::size_t k = std::min(batchSize, memories.size() - maxMemorySize);
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.3 ? 0.3 : exploringRate;
    learningSteps++;
    return;
}

void RL::SAC::save()
{
    actor.save("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].save(critiscName);
    }
    return;
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    for (int i = 0; i < QNET_NUM; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].load(critiscName);
        critics[i].copyTo(criticsTarget[i]);
    }
    return;
}
