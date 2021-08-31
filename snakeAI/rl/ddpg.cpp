#include "ddpg.h"

RL::DDPG::DDPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->alpha = 0.01;
    this->beta = 1;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->sa.resize(stateDim + actionDim);
    /* actor: a = P(s, theta) */
    this->actorP = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID, CROSS_ENTROPY);
    this->actorQ = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID, CROSS_ENTROPY);
    //this->actorP.softUpdateTo(actorQ, alpha);
    this->actorP.softUpdateTo(actorQ, alpha);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    this->criticMainNet = BPNN(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, true);
    this->criticTargetNet = BPNN(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, false);
    this->criticMainNet.softUpdateTo(criticTargetNet, alpha);
    return;
}

void RL::DDPG::perceive(Vec& state,
        Vec &action,
        Vec& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void RL::DDPG::setSA(Vec &state, Vec &Action)
{
    for (std::size_t i = 0; i < stateDim; i++) {
        sa[i] = state[i];
    }
    for (std::size_t i = stateDim; i < stateDim + actionDim; i++) {
        sa[i] = Action[i];
    }
    return;
}

int RL::DDPG::noiseAction(Vec &state)
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
    index = RL::max(out);
    return index;
}

int RL::DDPG::randomAction()
{
    return rand() % actionDim;
}

RL::Vec& RL::DDPG::greedyAction(Vec &state)
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

int RL::DDPG::action(Vec &state)
{
    return actorP.feedForward(state).argmax();
}

void RL::DDPG::experienceReplay(Transition& x)
{
    Vec cTarget(actionDim);
    Vec& p = actorP.output();
    Vec& q = actorQ.output();
    Vec& cTargetOutput = criticTargetNet.output();
    Vec& cMainOutput = criticMainNet.output();
    /* estimate action value */
    int i = RL::max(x.action);
    actorP.feedForward(x.state);
    setSA(x.state, p);
    criticMainNet.feedForward(sa);
    cTarget = cMainOutput;
    if (x.done == true) {
        cTarget[i] = x.reward;
    } else {
        actorQ.feedForward(x.nextState);
        setSA(x.nextState, q);
        criticTargetNet.feedForward(sa);
        int k = criticMainNet.feedForward(sa).argmax();
        cTarget[i] = x.reward + gamma * cTargetOutput[k];
    }
    /* update actorMainNet */
    Vec aTarget(q);
    aTarget[i] *= cTarget[i];
    actorP.gradient(x.state, aTarget);
    /* update criticMainNet */
    setSA(x.state, p);
    criticMainNet.gradient(sa, cTarget);
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
        /* update tagetNet */
        criticMainNet.softUpdateTo(criticTargetNet, alpha);
        learningSteps = 0;
    }
    if (learningSteps % 10 == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update tagetNet */
        actorP.softUpdateTo(actorQ, alpha);
    }
    /* experience replay */
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        experienceReplay(memories[k]);
    }
    actorP.optimize(optType, actorLearningRate);
    criticMainNet.optimize(optType, criticLearningRate);
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
    criticMainNet.save(criticPara);
    return;
}

void RL::DDPG::load(const std::string& actorPara, const std::string& criticPara)
{
    actorP.load(actorPara);
    actorP.softUpdateTo(actorQ, alpha);
    criticMainNet.load(criticPara);
    criticMainNet.softUpdateTo(criticTargetNet, alpha);
    return;
}

