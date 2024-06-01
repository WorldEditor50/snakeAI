#include "ddpg.h"

RL::DDPG::DDPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.99;
    beta = 1;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;
    /* actor: a = P(s, theta) */
    actorP = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Softmax::_(hiddenDim, actionDim, true));

    actorQ = BPNN(Layer<Tanh>::_(stateDim, hiddenDim,false),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  Layer<Tanh>::_(hiddenDim, hiddenDim,false),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  Softmax::_(hiddenDim, actionDim, false));
    actorP.copyTo(actorQ);
    /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
    criticP = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true),
                   LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                   LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    criticQ = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, false),
                   LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                   Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                   LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                   Layer<Sigmoid>::_(hiddenDim, actionDim, false));
    criticP.copyTo(criticQ);
}

void RL::DDPG::perceive(const Mat& state,
        const Mat &action,
        const Mat& nextState,
        float reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Mat& RL::DDPG::noiseAction(const Mat &state)
{
    actorQ.forward(state);
    Mat& out = actorQ.output();
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Random::engine);
    if (p < exploringRate) {
        for (std::size_t i = 0; i < actionDim; i++) {
            std::uniform_real_distribution<float> distribution(-1, 1);
            out[i] += distribution(Random::engine);
        }
    }
    return out;
}

RL::Mat& RL::DDPG::eGreedyAction(const Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Random::engine);
    if (p < exploringRate) {
        actorP.output().zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Random::engine);
        actorP.output()[index] = 1;
    } else {
        actorP.forward(state);
    }
    return actorP.output();
}

int RL::DDPG::action(const Mat &state)
{
    return actorP.show(), actorP.forward(state).argmax();
}

void RL::DDPG::experienceReplay(const Transition& x)
{
    Mat cTarget(actionDim, 1);
    /* estimate action value */
    Mat &p = actorP.forward(x.state);
    int i = p.argmax();
    Mat sa(stateDim + actionDim, 1);
    Mat::concat(0, sa, x.state, p);
    Mat &cMain = criticP.forward(sa);
    cTarget = cMain;
    if (x.done == true) {
        cTarget[i] = x.reward;
    } else {
        Mat &q = actorQ.forward(x.nextState);
        int j = q.argmax();
        Mat::concat(0, sa, x.nextState, q);
        Mat &cTargetOutput = criticQ.forward(sa);
        cTarget[i] = x.reward + gamma * cTargetOutput[j];
    }
    /* update criticMainNet */
    Mat::concat(0, sa, x.state, p);
    Mat &critic = criticP.forward(sa);
    criticP.backward(Loss::MSE(critic, cTarget));
    criticP.gradient(sa, cTarget);
    /* update actorMainNet */
    Mat &actor = actorP.forward(x.state);
    Mat::concat(0, sa, x.state, actor);
    criticP.forward(sa);
    Mat adv(actor);
    for (std::size_t k = 0; k < actionDim; k++) {
        adv[k] =  critic[k] - actor[k]*std::log(actor[k]/critic[k]);
    }
    Mat& actorOut = actorP.forward(x.state);
    actorP.backward(Loss::CrossEntropy(actorOut, adv));
    actorP.gradient(x.state, adv);
    return;
}

void RL::DDPG::learn(std::size_t maxMemorySize,
                     std::size_t replaceTargetIter,
                     std::size_t batchSize,
                     float actorLearningRate,
                     float criticLearningRate)
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
    actorP.optimize(OPT_NORMRMSPROP, actorLearningRate, 0.1);
    criticP.optimize(OPT_NORMRMSPROP, criticLearningRate, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
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
