#include "ppo.h"

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_)
{
    gamma = 0.99;
    beta = 0.5;
    delta = 0.01;
    epsilon = 0.2;
    exploringRate = 1;
    learningSteps = 0;
    stateDim = stateDim_;
    actionDim = actionDim_;

    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.04*std::log(0.04);

    actorP = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Softmax::_(hiddenDim, actionDim, true));

    actorQ = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, false),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                  Softmax::_(hiddenDim, actionDim, false));

    critic = BPNN(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                  TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                  Layer<Linear>::_(hiddenDim, actionDim, true));

}

RL::Mat &RL::PPO::eGreedyAction(const Mat &state)
{
    Mat& out = actorQ.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Mat &RL::PPO::noiseAction(const RL::Mat &state)
{
    Mat& out = actorQ.forward(state);
    return noise(out, exploringRate);
}

RL::Mat &RL::PPO::gumbelMax(const RL::Mat &state)
{
    Mat& out = actorQ.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Mat &RL::PPO::action(const Mat &state)
{
    return actorP.forward(state);
}

void RL::PPO::learnWithKLpenalty(std::vector<RL::Step> &trajectory, float learningRate)
{
    if (learningSteps % 16 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        learningSteps = 0;
    }
    /* reward */
    int end = trajectory.size() - 1;
    Mat criticState(stateDim + actionDim, 1);
    Mat::concat(0, criticState,
                trajectory[end].state,
                trajectory[end].action);
    float r = critic.forward(criticState).max();
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }

    float KLexpect = 0;
    for (int t = end; t >= 0; t--) {
        int k = trajectory[t].action.argmax();
        /* advangtage */
        Mat::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        Mat& v = critic.forward(criticState);
        float advantage = trajectory[t].reward - v[k];
        /* critic */
        Mat r = v;
        if (t == end) {
            r[k] = trajectory[t].reward;
        } else {
            Mat::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            v = critic.forward(criticState);
            r[k] = trajectory[t].reward + 0.99*v[k];
        }
        critic.gradient(criticState, r, Loss::MSE);
        /* actor */
        Mat& q = trajectory[t].action;
        Mat& p = actorP.forward(trajectory[t].state);
        float kl = p[k] * std::log(p[k]/q[k] + 1e-9);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        q[k] *= ratio*advantage - beta*kl;
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
    actorP.optimize(OPT_NORMRMSPROP, learningRate, 0.1);
    critic.optimize(OPT_NORMRMSPROP, 1e-3, 0.01);
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.01 ? 0.01 : exploringRate;
    learningSteps++;
    return;
}

void RL::PPO::learnWithClipObjective(std::vector<RL::Step> &trajectory, float learningRate)
{
    if (learningSteps % 16 == 0) {
        actorP.softUpdateTo(actorQ, 0.01);
        learningSteps = 0;
    }
    int end = trajectory.size() - 1;
    Mat criticState(stateDim + actionDim, 1);
    Mat::concat(0, criticState,
                trajectory[end].state,
                trajectory[end].action);
    float r = critic.forward(criticState).max();
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    for (int t = end; t >= 0; t--) {
        int k = trajectory[t].action.argmax();
        Mat::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        /* advangtage */
        Mat& v = critic.forward(criticState);
        float adv = trajectory[t].reward - v[k];
        /* critic */
        Mat r = v;
        if (t == end) {
            r[k] = trajectory[t].reward;
        } else {
            Mat::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            v = critic.forward(criticState);
            r[k] = trajectory[t].reward + 0.99*v[k];
        }
        critic.gradient(criticState, r, Loss::MSE);
        /* temperture parameter */
        Mat& q = trajectory[t].action;
        alpha.g[k] += -q[k]*std::log(q[k] + 1e-8) - entropy0;
        /* actor */
        Mat& p = actorP.forward(trajectory[t].state);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] *= ratio * adv;
        actorP.gradient(trajectory[t].state, q, Loss::CrossEntropy);
    }
    actorP.optimize(OPT_NORMRMSPROP, learningRate, 0.1);
    actorP.clamp(-1, 1);
    critic.optimize(OPT_NORMRMSPROP, 1e-3, 0.01);

    alpha.RMSProp(0.9, 1e-4, 0);
    alpha.clamp(0.2, 1.25);
#if 1
    std::cout<<"alpha:";
    alpha.val.show();
#endif
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.01 ? 0.01 : exploringRate;
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
