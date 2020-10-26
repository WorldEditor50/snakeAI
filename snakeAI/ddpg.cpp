#include "ddpg.h"

ML::DDPG::DDPG(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim)
{
    if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
        return;
    }
    this->gamma = 0.99;
    this->alpha = 0.01;
    this->beta = 1;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->sa.resize(stateDim + actionDim);
    /* actor: a = P(s, theta) */
    this->actorP = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
    this->actorQ = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
    //this->actorP.softUpdateTo(actorQ, alpha);
    this->actorP.softUpdateTo(actorQ, alpha);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    this->criticMainNet = MLP(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 1);
    this->criticTargetNet = MLP(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 0);
    this->criticMainNet.softUpdateTo(criticTargetNet, alpha);
    return;
}

void ML::DDPG::perceive(std::vector<double>& state,
        std::vector<double> &action,
        std::vector<double>& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

void ML::DDPG::setSA(std::vector<double> &state, std::vector<double> &Action)
{
    for (int i = 0; i < stateDim; i++) {
        sa[i] = state[i];
    }
    for (int i = stateDim; i < stateDim + actionDim; i++) {
        sa[i] = Action[i];
    }
    return;
}

int ML::DDPG::noiseAction(std::vector<double> &state)
{
    int index = 0;
    actorP.feedForward(state);
    std::vector<double>& out = actorP.getOutput();
    double p = double(rand() % 10000) / 10000;
    if (p < exploringRate) {
        for (int i = 0; i < actionDim; i++) {
            out[i] += double(rand() % 100 - rand() % 100) / 1000;
        }
    }
    index = maxQ(out);
    return index;
}

int ML::DDPG::randomAction()
{
    return rand() % actionDim;
}

std::vector<double>& ML::DDPG::greedyAction(std::vector<double> &state)
{
    double p = double(rand() % 10000) / 10000;
    std::vector<double> &out = actorP.getOutput();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        int index = rand() % actionDim;
        out[index] = 1;
    } else {
        actorP.feedForward(state);
    }
    return out;
}

int ML::DDPG::action(std::vector<double> &state)
{
    return actorP.feedForward(state);
}

int ML::DDPG::maxQ(std::vector<double>& q_value)
{
    int index = 0;
    double maxValue = q_value[0];
    for (int i = 0; i < q_value.size(); i++) {
        if (maxValue < q_value[i]) {
            maxValue = q_value[i];
            index = i;
        }
    }
    return index;
}

void ML::DDPG::experienceReplay(Transition& x)
{
    std::vector<double> cTarget(actionDim);
    std::vector<double>& p = actorP.getOutput();
    std::vector<double>& q = actorQ.getOutput();
    std::vector<double>& cTargetOutput = criticTargetNet.getOutput();
    std::vector<double>& cMainOutput = criticMainNet.getOutput();
    /* estimate action value */
    int i = maxQ(x.action);
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
        criticMainNet.feedForward(sa);
        int k = maxQ(cMainOutput);
        cTarget[i] = x.reward + gamma * cTargetOutput[k];
    }
    /* update actorMainNet */
    actorP.feedForward(x.state);
    actorQ.feedForward(x.state);
    double ratio = p[i] / (q[i] + 1e-9);
    double advangtage = cTarget[i] - cMainOutput[i];
    double epsilon = 0.2;
    if (advangtage > 0) {
        if (ratio > 1 + epsilon) {
            ratio = 1 + epsilon;
        }
    } else {
        if (ratio < 1 - epsilon) {
            ratio = 1 - epsilon;
        }
    }
    q[i] = 0.01 * q[i] + ratio * advangtage;
    actorP.gradient(x.state, q);
    /* update criticMainNet */
    setSA(x.state, p);
    criticMainNet.gradient(sa, cTarget);
    return;
}

void ML::DDPG::learn(int optType,
                     int maxMemorySize,
                     int replaceTargetIter,
                     int batchSize,
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
    for (int i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        experienceReplay(memories[k]);
    }
    actorP.optimize(optType, actorLearningRate);
    criticMainNet.optimize(optType, criticLearningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        int k = memories.size() / 3;
        for (int i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void ML::DDPG::save(const std::string& actorPara, const std::string& criticPara)
{
    actorP.save(actorPara);
    criticMainNet.save(criticPara);
    return;
}

void ML::DDPG::load(const std::string& actorPara, const std::string& criticPara)
{
    actorP.load(actorPara);
    actorP.softUpdateTo(actorQ, alpha);
    criticMainNet.load(criticPara);
    criticMainNet.softUpdateTo(criticTargetNet, alpha);
    return;
}

