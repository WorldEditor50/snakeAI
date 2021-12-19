#include "ppo.h"

RL::PPO::PPO(int stateDim, int hiddenDim, int actionDim)
{
    this->gamma = 0.99;
    this->beta = 0.5;
    this->delta = 0.01;
    this->epsilon = 0.2;
    this->exploringRate = 1;
    this->learningSteps = 0;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->actorP = BPNN(BPNN::Layers{
                            Layer<Tanh>::_(stateDim, hiddenDim, true),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                            SoftmaxLayer::_(hiddenDim, actionDim, true)
                        });
    this->actorQ = BPNN(BPNN::Layers{
                            Layer<Tanh>::_(stateDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                            SoftmaxLayer::_(hiddenDim, actionDim, false)
                        });
    this->critic = BPNN(BPNN::Layers{
                            Layer<Tanh>::_(stateDim, hiddenDim, true),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                            Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                            LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                            Layer<Sigmoid>::_(hiddenDim, actionDim, true)
                        });
    return;
}

RL::Vec &RL::PPO::sample(const Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        actorQ.output().assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        actorQ.output()[index] = 1;
    } else {
        actorQ.feedForward(state);
    }
    return actorQ.output();
}

RL::BPNN &RL::PPO::action(const Vec &state)
{
    return actorP.feedForward(state);
}

void RL::PPO::learnWithKLpenalty(OptType optType, double learningRate, std::vector<RL::Transition> &x)
{
    /* reward */
    int end = x.size() - 1;
    double r = RL::max(critic.feedForward(x[end].state).output());
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
    for (std::size_t t = 0; t < x.size(); t++) {
        int k = RL::argmax(x[t].action);
        /* advangtage */
        Vec& v = critic.feedForward(x[t].state).output();
        double advantage = x[t].reward - v[k];
        /* critic */
        Vec rt(actionDim, 0);
        rt[k] = x[t].reward;
        critic.gradient(x[t].state, rt, Loss::MSE);
        /* actor */
        Vec& q = x[t].action;
        Vec& p = actorP.feedForward(x[t].state).output();
        double kl = p[k] * log(p[k]/q[k]);
        q[k] += p[k] / q[k] * advantage - beta * kl;
        KLexpect += kl;
        actorP.gradient(x[t].state, q, Loss::CROSS_EMTROPY);
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
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    for (std::size_t t = 0; t < x.size(); t++) {
        int k = RL::argmax(x[t].action);
        /* advangtage */
        Vec& v = critic.feedForward(x[t].state).output();
        double adv = x[t].reward - v[k];
        /* critic */
        Vec rt(actionDim, 0);
        rt = v;
        if (t < x.size() - 1) {
            Vec vn = critic.feedForward(x[t + 1].state).output();
            rt[k] = x[t].reward + gamma*vn[k];
        } else {
            rt[k] = x[t].reward;
        }
        critic.gradient(x[t].state, rt, Loss::MSE);
        /* actor */
        Vec& q = x[t].action;
        Vec& p = actorP.feedForward(x[t].state).output();
        double ratio = p[k]/q[k];
        ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] += ratio * adv;
        actorP.gradient(x[t].state, q, Loss::CROSS_EMTROPY);
    }
    actorP.optimize(optType, learningRate);
    critic.optimize(optType, 0.001);
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
