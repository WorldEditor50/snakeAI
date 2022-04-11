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

    actorP = LstmNet(LSTM(stateDim, hiddenDim, hiddenDim, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                     LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                     SoftmaxLayer::_(hiddenDim, actionDim, true));
    actorP.lstm.ema = true;
    actorP.lstm.gamma = 0.5;

    actorQ = LstmNet(LSTM(stateDim, hiddenDim, hiddenDim, false),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                     LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                     SoftmaxLayer::_(hiddenDim, actionDim, false));

    critic = LstmNet(LSTM(stateDim, hiddenDim, hiddenDim, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                     LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                     Layer<Relu>::_(hiddenDim, actionDim, true));
    critic.lstm.ema = true;
    critic.lstm.gamma = 0.5;
}

RL::Vec &RL::PPO::eGreedyAction(const Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    if (p < exploringRate) {
        actorQ.output().assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        actorQ.output()[index] = 1;
    } else {
        actorQ.forward(state);
    }
    return actorQ.output();
}

RL::Vec &RL::PPO::action(const Vec &state)
{
    return actorP.forward(state);
}

void RL::PPO::learnWithKLpenalty(double learningRate, std::vector<RL::Transition> &trajectory)
{
    /* reward */
    int end = trajectory.size() - 1;
    double r = RL::max(critic.forward(trajectory[end].state));
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    double KLexpect = 0;
    std::vector<Vec> x;
    std::vector<Vec> y;
    std::vector<Vec> reward(trajectory.size(), Vec(actionDim, 0));
    for (std::size_t t = 0; t < trajectory.size(); t++) {
        int k = RL::argmax(trajectory[t].action);
        /* advangtage */
        Vec& v = critic.forward(trajectory[t].state);
        double advantage = trajectory[t].reward - v[k];
        /* critic */
        reward[t][k] = trajectory[t].reward;
        /* actor */
        Vec& q = trajectory[t].action;
        Vec& p = actorP.forward(trajectory[t].state);
        double kl = p[k] * log(p[k]/q[k]);
        q[k] += p[k] / q[k] * advantage - beta * kl;
        KLexpect += kl;
        x.push_back(trajectory[t].state);
        y.push_back(q);
    }
    /* KL-Penalty */
    KLexpect /= double(trajectory.size());
    if (KLexpect >= 1.5 * delta) {
        beta *= 2;
    } else if (KLexpect <= delta / 1.5) {
        beta /= 2;
    }
    actorP.forward(x);
    actorP.backward(x, y, Loss::CROSS_EMTROPY);
    actorP.optimize(learningRate, 0.01);
    critic.forward(x);
    critic.backward(x, reward, Loss::MSE);
    critic.optimize(0.001, 0.01);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::learnWithClipObjective(double learningRate, std::vector<RL::Transition> &trajectory)
{
    if (learningSteps % 10 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        //actorP.copyTo(actorQ);
        learningSteps = 0;
    }
    int end = trajectory.size() - 1;
    double r = RL::max(critic.forward(trajectory[end].state));
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    std::vector<Vec> x;
    std::vector<Vec> y;
    std::vector<Vec> reward(trajectory.size(), Vec(actionDim, 0));
    for (std::size_t t = 0; t < trajectory.size(); t++) {
        int k = RL::argmax(trajectory[t].action);
        /* advangtage */
        Vec& v = critic.forward(trajectory[t].state);
        double adv = trajectory[t].reward - v[k];
        /* critic */
        reward[t][k] = trajectory[t].reward;
        /* actor */
        Vec& q = trajectory[t].action;
        Vec& p = actorP.forward(trajectory[t].state);
        double ratio = p[k]/q[k];
        ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] += ratio * adv;
        x.push_back(trajectory[t].state);
        y.push_back(q);
    }
    actorP.forward(x);
    actorP.backward(x, y, Loss::CROSS_EMTROPY);
    actorP.optimize(learningRate, 0.01);
    critic.forward(x);
    critic.backward(x, reward, Loss::MSE);
    critic.optimize(0.001, 0.01);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::save(const std::string &actorPara, const std::string &criticPara)
{
    //actorP.save(actorPara);
    //critic.save(criticPara);
    return;
}

void RL::PPO::load(const std::string &actorPara, const std::string &criticPara)
{
    //actorP.load(actorPara);
    //actorP.copyTo(actorQ);
    //critic.load(criticPara);
    return;
}
