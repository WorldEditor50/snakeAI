#include "ddpg.h"

RL::DDPG::DDPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->beta = 1;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->sa = Vec(stateDim + actionDim, 0);
    /* actor: a = P(s, theta) */
    this->actorP = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                        LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                        Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                        LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                        SoftmaxLayer::_(hiddenDim, actionDim, true));

    this->actorQ = BPNN(Layer<Tanh>::_(stateDim, hiddenDim,false),
                        LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                        Layer<Tanh>::_(hiddenDim, hiddenDim,false),
                        LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                        SoftmaxLayer::_(hiddenDim, actionDim, false));
    this->actorP.copyTo(actorQ);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    this->criticP = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true),
                         LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                         Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                         LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                         Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    this->criticQ = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, false),
                         LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                         Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                         LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                         Layer<Sigmoid>::_(hiddenDim, actionDim, false));
    this->criticP.copyTo(criticQ);
}

void RL::DDPG::perceive(const Vec& state,
        const Vec &action,
        const Vec& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void RL::DDPG::setSA(const Vec &state, const Vec &actions)
{
    for (std::size_t i = 0; i < stateDim; i++) {
        sa[i] = state[i];
    }
    for (std::size_t i = stateDim; i < stateDim + actionDim; i++) {
        sa[i] = actions[i];
    }
    return;
}

RL::Vec& RL::DDPG::noiseAction(const Vec &state)
{
    actorQ.feedForward(state);
    Vec& out = actorQ.output();
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        for (std::size_t i = 0; i < actionDim; i++) {
            std::uniform_real_distribution<double> distribution(-1, 1);
            out[i] += distribution(Rand::engine);
        }
    }
    return out;
}

int RL::DDPG::randomAction()
{
    return rand() % actionDim;
}

RL::Vec& RL::DDPG::eGreedyAction(const Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        actorP.output().assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        actorP.output()[index] = 1;
    } else {
        actorP.feedForward(state);
    }
    return actorP.output();
}

int RL::DDPG::action(const Vec &state)
{
    return actorP.show(), actorP.feedForward(state).argmax();
}

void RL::DDPG::experienceReplay(Transition& x)
{
    Vec cTarget(actionDim);
    /* estimate action value */
    Vec &p = actorP.feedForward(x.state).output();
    int i = RL::argmax(p);
    setSA(x.state, p);
    Vec &cMain = criticP.feedForward(sa).output();
    cTarget = cMain;
    if (x.done == true) {
        cTarget[i] = x.reward;
    } else {
        Vec &q = actorQ.feedForward(x.nextState).output();
        int j = RL::argmax(q);
        setSA(x.nextState, q);
        Vec &cTargetOutput = criticQ.feedForward(sa).output();
        cTarget[i] = x.reward + gamma * cTargetOutput[j];
    }
    /* update criticMainNet */
    setSA(x.state, p);
    criticP.gradient(sa, cTarget, Loss::MSE);
    /* update actorMainNet */
    Vec &actor = actorP.feedForward(x.state).output();
    setSA(x.state, actor);
    Vec &critic = criticP.feedForward(sa).output();
    Vec adv(actor);
    for (std::size_t k = 0; k < actionDim; k++) {
        adv[k] =  critic[k] - actor[k]*log(actor[k]/critic[k]);
    }
    actorP.gradient(x.state, adv, Loss::CROSS_EMTROPY);
    return;
}

void RL::DDPG::learn(OptType optType,
                     std::size_t maxMemorySize,
                     std::size_t replaceTargetIter,
                     std::size_t batchSize,
                     double actorLearningRate,
                     double criticLearningRate)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update critic */
        criticP.softUpdateTo(criticQ, 0.01);
        learningSteps = 0;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update actor */
        actorP.softUpdateTo(actorQ, 0.01);
    }
    /* experience replay */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Rand::engine);
        experienceReplay(memories[k]);
    }
    actorP.optimize(optType, actorLearningRate, 0.1);
    criticP.optimize(optType, criticLearningRate, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
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
