#include "sac.h"
#include "layer.h"
#include "loss.h"

RL::SAC::SAC(size_t stateDim_, size_t hiddenDim, size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    annealing = ExpAnnealing(0.01, 0.12, 1e-4);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 =  RL::entropy(0.1);
    actor = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    for (int i = 0; i < max_qnet_num; i++) {
        critics[i] = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                      TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));
        criticsTarget[i] = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                            TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, true, false));
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
    return RL::gumbelSoftmax(out, alpha.val);
}

RL::Tensor& RL::SAC::action(const RL::Tensor &state)
{
    return actor.forward(state);
}

void RL::SAC::experienceReplay(const RL::Transition &x, float beta)
{
    /* train critic net */
#if 0
    {
        /* select action */
        const Tensor& nextProb = actor.forward(x.nextState);
        /* select value */;
        Tensor q(actionDim, 1);
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = critics[i].forward(x.state);
            q += qi;
        }
        q /= MAX_CRITICS;
        Tensor nextValue(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            nextValue[i] = nextProb[i]*(q[i] - alpha[i]*std::log(nextProb[i]));
        }
        int k = x.action.argmax();
        Tensor reward(actionDim, 1);
        reward[k] = x.reward;
        Tensor qTarget(actionDim, 1);
        for (int i = 0; i < actionDim; i++) {
            if (x.done) {
                qTarget[i] = reward[i];
            } else {
                qTarget[i] = reward[i] + gamma*nextValue[i];
            }
        }
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor &out = critics[i].forward(x.state);
            critics[i].backward(Loss::MSE::df(out, qTarget));
            critics[i].gradient(x.state, qTarget);
        }
    }
#else
    {
        /* select action */
        const Tensor& nextProb = actor.forward(x.nextState);
        std::size_t k = nextProb.argmax();
        /* select value */;
        float q = 0;
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = criticsTarget[i].forward(x.nextState);
            q += qi[k];
        }
        q /= max_qnet_num;
        float nextValue = 0;
        //nextValue = (nextProb[k] + beta)*(q - alpha[k]*std::log(nextProb[k] + beta));
        nextValue = Metrics::KL(nextProb[k] + beta, std::exp(q))*alpha[k];
        float qTarget = 0;
        if (x.done) {
            qTarget = x.reward;
        } else {
            qTarget = x.reward + gamma*nextValue;
        }
        k = x.action.argmax();
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor &out = critics[i].forward(x.state);
            Tensor p = out;
            p[k] = qTarget;
            critics[i].backward(Loss::MSE::df(out, p));
            critics[i].gradient(x.state, p);
        }
    }
#endif
    /* train policy net */
    {
        const Tensor& p = actor.forward(x.state);
        Tensor loss(actionDim, 1);
        Tensor q(actionDim, 1);
        for (int i = 0; i < max_qnet_num; i++) {
            const Tensor& qi = critics[i].forward(x.state);
            q += qi;
        }
        q /= max_qnet_num;
        for (int i = 0; i < actionDim; i++) {
            /*
                L(p) = KL(p||e^q) = -p*log(p/e^q) = -p*log(p) + p*q
                L(p) = p*q - alpha*p*log(p))
                time shift loss:
                        focus on current policy
                        e^(beta*D)[L] = Σ(beta*D)^n/n!)L(p) = L(p + beta)
            */
            //float dL = (p[i] + beta)*(q - alpha[i]*std::log(p[i] + beta));
            float dL = Metrics::KL(p[i] + beta, std::exp(q[i]))*alpha[i];
            loss[i] = p[i] - dL;
        }
        int k = p.argmax();
        float advantage = 0;
        advantage = beta*(1.0/(1 - 0.5*x.reward) - q[k]);
        loss[k] += advantage;
        actor.backward(loss);
        actor.gradient(x.state, loss);
    }

    /* alpha */
    {
        const Tensor& prob = x.action;
        for (int i = 0; i < actionDim; i++) {
            alpha.g[i] += (RL::entropy(prob[i]) - entropy0)*alpha[i];
        }
    }
    return;
}

void RL::SAC::learn(size_t maxMemorySize, size_t replaceTargetIter, size_t batchSize, float learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }

    if (learningSteps % replaceTargetIter == 0) {
        /* update */
        for (int i = 0; i < max_qnet_num; i++) {
            critics[i].softUpdateTo(criticsTarget[i], (i + 1)*1e-4);
        }
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> uniform(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = uniform(Random::engine);
        float beta = float(k)/maxMemorySize;
        experienceReplay(memories[k], beta);
    }
    actor.RMSProp(1e-2, 0.9, 0);
#if 1
    std::cout<<"annealing:"<<annealing.val<<",alpha:";
    alpha.val.printValue();
#endif
    alpha.RMSProp(1e-5, 0.9, 0);
    for (int i = 0; i < max_qnet_num; i++) {
        critics[i].RMSProp(1e-3, 0.9, 0);
    }
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size()/16;
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
    for (int i = 0; i < max_qnet_num; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].save(critiscName);
    }
    return;
}

void RL::SAC::load()
{
    actor.load("sac_actor");
    for (int i = 0; i < max_qnet_num; i++) {
        std::string critiscName = std::string("sac_critic_") + std::to_string(i);
        critics[i].load(critiscName);
        critics[i].copyTo(criticsTarget[i]);
    }
    return;
}
