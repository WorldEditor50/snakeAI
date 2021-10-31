#include "ddpg.h"

RL::DDPG::DDPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->beta = 1;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->sa.resize(stateDim + actionDim);
    /* actor: a = P(s, theta) */
    this->actorP = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID, CROSS_ENTROPY);
    this->actorQ = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID, CROSS_ENTROPY);
    this->actorP.copyTo(actorQ);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    this->criticP = BPNN(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID, MSE);
    this->criticQ = BPNN(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID, MSE);
    this->criticP.copyTo(criticQ);
    return;
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

int RL::DDPG::noiseAction(const Vec &state)
{
    int index = 0;
    actorP.feedForward(state);
    Vec& out = actorP.output();
    double p = double(rand() % 10000) / 10000;
    if (p < exploringRate) {
        for (std::size_t i = 0; i < actionDim; i++) {
            out[i] += double(rand() % 100 - rand() % 100) / 1000;
        }
    }
    index = RL::argmax(out);
    return index;
}

int RL::DDPG::randomAction()
{
    return rand() % actionDim;
}

RL::Vec& RL::DDPG::greedyAction(const Vec &state)
{
    double p = double(rand() % 10000) / 10000;
    Vec &out = actorP.output();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        int index = rand() % actionDim;
        out[index] = 1;
    } else {
        actorP.feedForward(state);
    }
    return out;
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
    /* update actorMainNet */
    actorP.gradient(x.state, cMain);
    /* update criticMainNet */
    setSA(x.state, p);
    criticP.gradient(sa, cTarget);
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
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        experienceReplay(memories[k]);
    }
    actorP.optimize(optType, actorLearningRate);
    criticP.optimize(optType, criticLearningRate);
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
