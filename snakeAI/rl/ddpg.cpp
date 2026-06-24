#include "ddpg.h"
#include "layer.h"
#include "loss.h"

RL::DDPG::DDPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_),
     gamma(0.99), exploringRate(1), learningSteps(0)
{
    /* actor: π(s) → softmax action probabilities (stochastic policy for exploration) */
    actorP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    actorQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, false));
    actorP.copyTo(actorQ);

    /* critic: Q(s) → [Q(s,a_0), ..., Q(s,a_{n-1})] */
    criticP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                  Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));

    criticQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                  Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
    criticP.copyTo(criticQ);
}

void RL::DDPG::perceive(const Tensor& state,
                        const Tensor &action,
                        const Tensor& nextState,
                        float reward,
                        bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Tensor& RL::DDPG::noiseAction(const Tensor &state)
{
    actorP.forward(state);
    Tensor& out = actorP.output();
    return noise(out);
}

RL::Tensor &RL::DDPG::gumbelMax(const RL::Tensor &state)
{
    actorP.forward(state);
    Tensor& out = actorP.output();
    return RL::gumbelSoftmax(out, exploringRate);
}

RL::Tensor& RL::DDPG::action(const Tensor &state)
{
    return actorP.forward(state);
}

void RL::DDPG::experienceReplay(const Transition& x)
{
    /* ================================================
     * 1. Critic Training:
     *    Q(s, a_taken) → r + γ * Q'(s', argmax π'(s'))
     *    using target networks for stability
     * ================================================ */
    {
        /* action actually taken in the transition */
        std::size_t actionTaken = x.action.argmax();

        /* current Q(s) for all actions */
        const Tensor& qCurrent = criticP.forward(x.state);

        /* compute TD target for Q(s, a_taken) */
        float tdTarget;
        if (x.done) {
            tdTarget = x.reward;
        } else {
            /* target policy selects best action on next state */
            const Tensor& nextPolicy = actorQ.forward(x.nextState);
            int bestNextAction = nextPolicy.argmax();
            /* target critic evaluates that action */
            const Tensor& nextQ = criticQ.forward(x.nextState);
            tdTarget = x.reward + gamma * nextQ[bestNextAction];
        }

        /* build Q-target: copy Q(s) and overwrite the taken action */
        Tensor qTarget = qCurrent;
        qTarget[actionTaken] = tdTarget;

        /* train critic: minimize MSE(Q(s), Q-target) */
        criticP.backward(x.state, Loss::MSE::df(qCurrent, qTarget));
    }

    /* ================================================
     * 2. Actor Training:
     *    maximize J(π) = Σ π_i(s) * Q_i(s)
     *    ∇w.r.t π_i: -Q_i (we minimize -J through backward)
     *    The softmax layer's Jacobian transforms this
     *    into the correct gradient on logits:
     *      dy_i = π_i * (Q_i - Σ π_j * Q_j)
     * ================================================ */
    {
        const Tensor& policy = actorP.forward(x.state);
        const Tensor& qValues = criticP.forward(x.state);

        /* policy gradient: d(-ΣπQ)/dπ_i = -Q_i */
        Tensor err(actionDim, 1);
        for (std::size_t i = 0; i < actionDim; i++) {
            err[i] = -qValues[i];
        }

        actorP.backward(x.state, err);
    }

    return;
}

void RL::DDPG::learn(std::size_t maxMemorySize,
                     std::size_t replaceTargetIter,
                     std::size_t batchSize)
{
    if (memories.size() < batchSize) {
        return;
    }

    /* Polyak averaging of target networks (every step for stable learning) */
    float tau = 5e-3;
    criticP.softUpdateTo(criticQ, tau);
    actorP.softUpdateTo(actorQ, tau);

    /* experience replay with random mini-batch */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Random::engine);
        experienceReplay(memories[k]);
    }

    /* Apply gradient updates with Adam optimizer */
    actorP.Adam(1e-3, 0.99, 0.9, 1e-4);
    criticP.Adam(1e-3, 0.99, 0.9, 1e-4);

    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }

    /* annealing exploration rate */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.2 ? 0.2 : exploringRate;
    learningSteps++;
    return;
}

void RL::DDPG::save(const std::string& actorPara, const std::string& criticPara)
{
    actorP.save(actorPara);
    criticP.save(criticPara);
    return;
}

void RL::DDPG::load(const std::string& actorPara, const std::string& criticPara)
{
    actorP.load(actorPara);
    actorP.copyTo(actorQ);
    criticP.load(criticPara);
    criticP.copyTo(criticQ);
    return;
}


