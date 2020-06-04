#include "ppo.h"

void ML::PPO::CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                        int maxMemorySize , int replaceTargetIter, int batchSize)
{
    if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
            maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
        return;
    }
    this->gamma = 0.99;
    this->exploringRate = 1;
    this->epsilon = 0.2;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->maxMemorySize = maxMemorySize;
    this->replaceTargetIter = replaceTargetIter;
    this->learningStep = 0;
    this->batchSize = batchSize;
    this->actor.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_KL);
    this->actorPrime.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_KL);
    this->critic.CreateNet(stateDim, hiddenDim, hiddenLayerNum, 1, 1, ACTIVATE_RELU, LOSS_MSE);
    return;
}

int ML::PPO::GreedyAction(std::vector<double> &state)
{
    double p = (rand() % 10000) / 10000;
    int index = 0;
    if (p < exploringRate) {
        std::vector<double>& action = actor.GetOutput();
        for (int i = 0; i < actionDim; i++) {
            action[i] = (rand() % 10000) / 10000;
        }
        index = actor.Argmax();
    } else {
        index = actor.FeedForward(state);
    }
    return index;
}

int ML::PPO::Action(std::vector<double> &state)
{
    return actor.FeedForward(state);
}

void ML::PPO::Perceive(std::vector<Step>& trajectory)
{
    double r = 0;
    for (int i = trajectory.size() - 1; i >= 0; i--) {
        r = gamma * r + trajectory[i].reward;
        trajectory[i].reward = r;
    }
    memories.push_back(trajectory);
    return;
}

void ML::PPO::ExperienceReplay(std::vector<Step>& trajectory)
{
    std::vector<double>& v = critic.GetOutput();
    std::vector<double>& p = actor.GetOutput();
    std::vector<double>& pp = actorPrime.GetOutput();
    for (int i = 0; i < trajectory.size(); i++) {
        /* actor */
        /* advangtage */
        critic.FeedForward(trajectory[i].state);
        double adv = trajectory[i].reward - v[0];
        /* ratio */
        actor.FeedForward(trajectory[i].state);
        actorPrime.FeedForward(trajectory[i].state);
        std::vector<double> ratio(actionDim, 0);
        for (int j = 0; j < actionDim; j++) {
            ratio[j] = pp[j] * adv;
#if 0
            double ratioTmp = p[j] / (pp[j] + 1e5);
            /* clip */
            if (adv > 0) {
                ratioTmp = (1 + epsilon) * adv;
            } else {
                ratioTmp = (1 - epsilon) * adv;
            }
            if (ratio[j] > ratioTmp) {
                ratio[j] = ratioTmp;
            }
#endif
        }
        actor.Gradient(trajectory[i].state, ratio);
        /* critic */
        std::vector<double> discounted_r(1);
        discounted_r[0] = trajectory[i].reward;
        critic.Gradient(trajectory[i].state, discounted_r);
    }
    return;
}

void ML::PPO::Learn(int optType, double learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningStep % replaceTargetIter == 0) {
        std::cout<<"update"<<std::endl;
        /* update */
        actor.CopyTo(actorPrime);
        learningStep = 0;
    }
    /* experience replay */
    for (int i = 0; i < batchSize; i++) {
        int k = rand() % memories.size();
        ExperienceReplay(memories[k]);
    }
    actor.Optimize(optType, learningRate);
    critic.Optimize(optType, learningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        int k = memories.size() / 4;
        for (int i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate = exploringRate * 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningStep++;
    return;
}

void ML::PPO::Save(const std::string &actorPara, const std::string &criticPara)
{
    actor.Save(actorPara);
    critic.Save(criticPara);
    return;
}

void ML::PPO::Load(const std::string &actorPara, const std::string &criticPara)
{
    actor.Load(actorPara);
    actor.CopyTo(actorPrime);
    critic.Load(criticPara);
    return;
}
