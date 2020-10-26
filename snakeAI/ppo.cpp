#include "ppo.h"

ML::PPO::PPO(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim)
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
    this->actorP = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
    this->actorQ = MLP(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
    this->critic = MLP(stateDim, hiddenDim, hiddenLayerNum, 1, 1, ACTIVATE_RELU, LOSS_MSE);
    return;
}

int ML::PPO::greedyAction(std::vector<double> &state)
{
    std::vector<double>& out = actorQ.getOutput();
    double p = double(rand() % 10000) / 10000;
    int index = 0;
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        index = rand() % actionDim;
        out[index] = 1;
    } else {
        index = actorQ.feedForward(state);
    }
    return index;
}

int ML::PPO::action(std::vector<double> &state)
{
    return actorP.feedForward(state);
}

double ML::PPO::KLmean(std::vector<double> &p, std::vector<double> &q)
{
    double sum = 0;
    double n = 0;
    for (int i = 0; i < p.size(); i++) {
        sum += p[i] * (log(p[i]) - log(q[i] + 1e-5));
        n++;
    }
    sum = sum / n;
    return sum;
}

double ML::PPO::getValue(std::vector<double> &s)
{
    std::vector<double>& v = critic.getOutput();
    critic.feedForward(s);
    return v[0];
}

void ML::PPO::learnWithKLpenalty(int optType, double learningRate, std::vector<ML::Transit> &x)
{
    /* reward */
    int end = x.size() - 1;
    double r = getValue(x[end].state);
    for (int i = end; i >= 0; i--) {
        r = x[i].reward + gamma * r;
        x[i].reward = r;
    }
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
    }
    double n = 0;
    double KLexpect = 0;
    std::vector<double>& v = critic.getOutput();
    std::vector<double>& p = actorP.getOutput();
    for (int i = 0; i < x.size(); i++) {
        /* advangtage */
        critic.feedForward(x[i].state);
        double advantage = x[i].reward - v[0];
        /* critic */
        std::vector<double> discounted_r(1);
        discounted_r[0] = x[i].reward;
        critic.gradient(x[i].state, discounted_r);
        /* actor */
        std::vector<double>& q = x[i].action;
        actorP.feedForward(x[i].state);
        std::vector<double> y(q);
        int k = maxAction(q);
        double kl = p[k] * (log(p[k]) - log(q[k] + 1e-9));
        y[k] = q[k] + p[k] / (q[k] + 1e-9) * advantage - beta * kl;
        KLexpect += kl;
        n++;
        actorP.gradient(x[i].state, y);
    }
    /* KL-Penalty */
    KLexpect = KLexpect / n;
    if (KLexpect >= 1.5 * delta) {
        beta *= 2;
    } else if (KLexpect <= delta / 1.5) {
        beta /= 2;
    }
    actorP.optimize(optType, learningRate);
    critic.optimize(optType, 0.0001);
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void ML::PPO::learnWithClipObject(int optType, double learningRate, std::vector<ML::Transit> &x)
{
    /* reward */
    int end = x.size() - 1;
    double r = getValue(x[end].state);
    for (int i = end; i >= 0; i--) {
        r = x[i].reward + gamma * r;
        x[i].reward = r;
    }
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    std::vector<double>& v = critic.getOutput();
    std::vector<double>& p = actorP.getOutput();
    for (int i = 0; i < x.size(); i++) {
        /* advangtage */
        critic.feedForward(x[i].state);
        double advantage = x[i].reward - v[0];
        /* critic */
        std::vector<double> discount_r(1);
        discount_r[0] = x[i].reward;
        critic.gradient(x[i].state, discount_r);
        /* actor */
        std::vector<double>& q = x[i].action;
        actorP.feedForward(x[i].state);
        int k = maxAction(q);
        double ratio = p[k] / (q[k] + 1e-9);
        if (advantage > 0) {
            if (ratio > 1 + epsilon) {
                ratio = 1 + epsilon;
            }
        } else {
            if (ratio < 1 - epsilon) {
                ratio = 1 - epsilon;
            }
        }
        q[k] = q[k] + ratio * advantage;
        actorP.gradient(x[i].state, q);
    }
    actorP.optimize(optType, learningRate);
    critic.optimize(optType, 0.0001);
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

int ML::PPO::maxAction(std::vector<double>& value)
{
    int index = 0;
    double maxValue = value[0];
    for (int i = 0; i < value.size(); i++) {
        if (maxValue < value[i]) {
            maxValue = value[i];
            index = i;
        }
    }
    return index;
}

void ML::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    actorP.save(actorPara);
    critic.save(criticPara);
    return;
}

void ML::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    actorP.load(actorPara);
    actorP.copyTo(actorQ);
    critic.load(criticPara);
    return;
}
