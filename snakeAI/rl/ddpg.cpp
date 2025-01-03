#include "ddpg.h"
#include "layer.h"
#include "loss.h"

RL::DDPG::DDPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    beta = 1;
    /* actor: a = P(s, theta) */
    actorP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    actorQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                 LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, false));
    actorP.copyTo(actorQ);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
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
    Tensor& out = actorP.forward(state);
    return noise(out);
}

RL::Tensor &RL::DDPG::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actorP.forward(state);
    return RL::gumbelSoftmax(out, exploringRate);
}

RL::Tensor& RL::DDPG::action(const Tensor &state)
{
    return actorP.forward(state);
}

void RL::DDPG::experienceReplay(const Transition& x)
{
    /* train critic */
    {
        std::size_t i = x.action.argmax();
        Tensor ct = criticP.forward(x.state);
        if (x.done == true) {
            ct[i] = x.reward;
        } else {
            Tensor &aq = actorQ.forward(x.nextState);
            int k = aq.argmax();
            Tensor &cq = criticQ.forward(x.nextState);
            ct[k] = x.reward + gamma*cq[k];
        }
        Tensor &cp = criticP.forward(x.state);
        criticP.backward(Loss::MSE::df(cp, ct));
        criticP.gradient(x.state, ct);
    }

    /* train actor */
    {
        Tensor &p = actorP.forward(x.state);
        Tensor& q = criticP.forward(x.state);
        Tensor dLoss(actionDim, 1);
        for (std::size_t i = 0; i < actionDim; i++) {
            dLoss[i] = p[i] - p[i]*q[i];
        }
        actorP.backward(dLoss);
        actorP.gradient(x.state, dLoss);
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
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update critic */
        criticP.softUpdateTo(criticQ, 0.01);
        learningSteps = 0;
        std::cout<<"update target net"<<std::endl;
        /* update actor */
        actorP.softUpdateTo(actorQ, 0.01);
    }
    /* experience replay */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Random::engine);
        experienceReplay(memories[k]);
    }
    actorP.RMSProp(0.01, 0.9, 0.01);
    criticP.RMSProp(1e-3);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
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
