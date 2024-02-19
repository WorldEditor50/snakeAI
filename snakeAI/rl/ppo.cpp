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

    actorP = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  SoftmaxLayer::_(hiddenDim, actionDim, true));

    actorQ = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, false),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  SoftmaxLayer::_(hiddenDim, actionDim, false));

    critic = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Linear>::_(hiddenDim, actionDim, true));

}

RL::Mat &RL::PPO::eGreedyAction(const Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        actorQ.output().zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        actorQ.output()[index] = 1;
    } else {
        actorQ.forward(state);
    }
    return actorQ.output();
}

RL::Mat &RL::PPO::action(const Mat &state)
{
    return actorP.forward(state);
}

void RL::PPO::learnWithKLpenalty(float learningRate, std::vector<RL::Step> &trajectory)
{
    /* reward */
    int end = trajectory.size() - 1;
    float r = critic.forward(trajectory[end].state).max();
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    if (learningSteps % 16 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    float KLexpect = 0;
    for (std::size_t t = 0; t < trajectory.size(); t++) {
        int k = trajectory[t].action.argmax();
        /* advangtage */
        Mat& v = critic.forward(trajectory[t].state);
        float advantage = trajectory[t].reward - v[k];
        /* critic */
        Mat r(actionDim, 1);
        r[k] = trajectory[t].reward;
        critic.gradient(trajectory[t].state, r, Loss::MSE);
        /* actor */
        Mat& q = trajectory[t].action;
        Mat& p = actorP.forward(trajectory[t].state);
        float kl = p[k] * std::log(p[k]/q[k] + 1e-9);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        q[k] += ratio*advantage - beta*kl;
        KLexpect += kl;
        actorP.gradient(trajectory[t].state, q, Loss::CrossEntropy);
    }
    /* KL-Penalty */
    KLexpect /= float(trajectory.size());
    if (KLexpect >= 1.5 * delta) {
        beta *= 2;
    } else if (KLexpect <= delta / 1.5) {
        beta /= 2;
    }
    actorP.optimize(OPT_RMSPROP, learningRate, 0.1);
    critic.optimize(OPT_RMSPROP, 1e-3, 0.01);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::learnWithClipObjective(float learningRate, std::vector<RL::Step> &trajectory)
{
    if (learningSteps % 16 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        learningSteps = 0;
    }
    int end = trajectory.size() - 1;
    float r = critic.forward(trajectory[end].state).max();
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    for (int t = end; t >= 0; t--) {
        int k = trajectory[t].action.argmax();
        /* advangtage */
        Mat& v = critic.forward(trajectory[t].state);
        float adv = trajectory[t].reward - v[k];
        /* critic */
        Mat r = v;
        if (t == end) {
            r[k] = trajectory[t].reward;
        } else {
            v = critic.forward(trajectory[t + 1].state);
            r[k] = trajectory[t].reward + 0.99*v[k];
        }
        critic.gradient(trajectory[t].state, r, Loss::MSE);
        /* actor */
        Mat& q = trajectory[t].action;
        Mat& p = actorP.forward(trajectory[t].state);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] += ratio * adv;
        actorP.gradient(trajectory[t].state, q, Loss::CrossEntropy);
    }
    actorP.optimize(OPT_RMSPROP, learningRate, 0.1);
    actorP.clamp(-1, 1);
    critic.optimize(OPT_RMSPROP, 1e-3, 0.01);

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
