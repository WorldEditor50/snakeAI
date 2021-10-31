#include "ppo.h"

RL::PPO::PPO(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim)
{
    if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1) {
        return;
    }
    this->gamma = 0.99;
    this->beta = 0.5;
    this->delta = 0.01;
    this->epsilon = 0.2;
    this->exploringRate = 1;
    this->learningSteps = 0;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->actorP = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, true, SIGMOID, CROSS_ENTROPY);
    this->actorQ = BPNN(stateDim, hiddenDim, hiddenLayerNum, actionDim, false, SIGMOID, CROSS_ENTROPY);
    this->critic = BPNN(stateDim, hiddenDim, 2, 1, 1, RELU, MSE);
    return;
}

void RL::PPO::continousAction(const Vec &state, Vec &act)
{
    double mu = RL::mean(state);
    double sigma = RL::variance(state);
    double sup = 0;
    double inf = 10;
    sigma = sqrt(sigma);
    RL::normalDistribution(mu, sigma, sup, inf, act, actionDim);
    return;
}

RL::BPNN &RL::PPO::greedyAction(const Vec &state)
{
    Vec& out = actorQ.output();
    double p = double(rand() % 10000) / 10000;
    int index = 0;
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        index = rand() % actionDim;
        out[index] = 1;
    } else {
        index = actorQ.feedForward(state).argmax();
    }
    return actorQ;
}

RL::BPNN &RL::PPO::action(const Vec &state)
{
    return actorP.feedForward(state);
}

void RL::PPO::learnWithKLpenalty(OptType optType, double learningRate, std::vector<RL::Transition> &x)
{
    /* reward */
    int end = x.size() - 1;
    double r = critic.feedForward(x[end].state).output()[0];
    for (int i = end; i >= 0; i--) {
        r = x[i].reward + gamma * r;
        x[i].reward = r;
    }
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    double KLexpect = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        /* advangtage */
        Vec& v = critic.feedForward(x[i].state).output();
        double advantage = x[i].reward - v[0];
        /* critic */
        Vec discounted_r(1);
        discounted_r[0] = x[i].reward;
        critic.gradient(x[i].state, discounted_r);
        /* actor */
        Vec& q = x[i].action;
        Vec& p = actorP.feedForward(x[i].state).output();
        int k = RL::argmax(q);
        double kl = p[k] * log(p[k]/q[k]);
        q[k] += p[k] / q[k] * advantage - beta * kl;
        KLexpect += kl;
        actorP.gradient(x[i].state, q);
    }
    /* KL-Penalty */
    KLexpect /= double(x.size());
    if (KLexpect >= 1.5 * delta) {
        beta *= 2;
    } else if (KLexpect <= delta / 1.5) {
        beta /= 2;
    }
    actorP.optimize(optType, learningRate);
    critic.optimize(optType, 0.0001);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::learnWithClipObject(OptType optType, double learningRate, std::vector<RL::Transition> &x)
{
    /* reward */
    int end = x.size() - 1;
    double r = critic.feedForward(x[end].state).output()[0];
    for (int i = end; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        x[i].reward = r;
    }
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }

    for (std::size_t i = 0; i < x.size(); i++) {
        /* advangtage */
        Vec& v = critic.feedForward(x[i].state).output();
        double advantage = x[i].reward - v[0];
        /* critic */
        Vec discount_r(1);
        discount_r[0] = x[i].reward;
        critic.gradient(x[i].state, discount_r);
        /* actor */
        Vec& q = x[i].action;
        Vec& p = actorP.feedForward(x[i].state).output();
        int k = RL::argmax(q);
        double ratio = p[k]/q[k];
        if (advantage > 0) {
            ratio = (ratio > 1 + epsilon) ? (1 + epsilon) : ratio;
        } else {
            ratio = (ratio < 1 - epsilon) ? (1 - epsilon) : ratio;
        }
        //ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] += ratio * advantage;
        actorP.gradient(x[i].state, q);
    }
    actorP.optimize(optType, learningRate);
    critic.optimize(optType, 0.0001);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    actorP.save(actorPara);
    critic.save(criticPara);
    return;
}

void RL::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    actorP.load(actorPara);
    actorP.copyTo(actorQ);
    critic.load(criticPara);
    return;
}
