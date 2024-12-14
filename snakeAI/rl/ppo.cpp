#include "ppo.h"
#include "layer.h"
#include "loss.h"

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    beta = 0.5;
    delta = 0.01;
    epsilon = 0.2;
    learningSteps = 0;
    annealing = ExpAnnealing(0.01, 0.12);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.12*std::log(0.12);

    actorP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    actorQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, false));

    critic = Net(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Linear>::_(hiddenDim, actionDim, true, true));

}

RL::Tensor &RL::PPO::eGreedyAction(const Tensor &state)
{
    Tensor& out = actorQ.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor &RL::PPO::noiseAction(const RL::Tensor &state)
{
    Tensor& out = actorQ.forward(state);
    return noise(out, exploringRate);
}

RL::Tensor &RL::PPO::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actorQ.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::PPO::action(const Tensor &state)
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
    Tensor criticState(stateDim + actionDim, 1);
    Tensor::concat(0, criticState,
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
        Tensor::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        Tensor v = critic.forward(criticState);
        float advantage = trajectory[t].reward - v[k];
        /* critic */
        Tensor r = v;
        if (t == end) {
            r[k] = trajectory[t].reward;
        } else {
            Tensor::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            Tensor &v1 = critic.forward(criticState);
            r[k] = trajectory[t].reward + 0.99*v1[k];
        }
        critic.backward(Loss::MSE(v, r));
        critic.gradient(criticState, r);
        /* temperture parameter */
        Tensor& q = trajectory[t].action;
        alpha.g[k] += (-q[k]*std::log(q[k] + 1e-8) - entropy0)*alpha[k];
        /* actor */
        Tensor p = actorP.forward(trajectory[t].state);
        float kl = p[k] * std::log(p[k]/q[k] + 1e-9);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        q[k] *= ratio*advantage - beta*kl;
        KLexpect += kl;
        actorP.backward(Loss::CrossEntropy(p, q));
        actorP.gradient(trajectory[t].state, q);
    }
    /* KL-Penalty */
    KLexpect /= float(trajectory.size());
    if (KLexpect >= 1.5 * delta) {
        beta *= 2;
    } else if (KLexpect <= delta / 1.5) {
        beta /= 2;
    }
    actorP.RMSProp(learningRate, 0.9, annealing.step());
    critic.RMSProp(1e-3, 0.9, 0.01);
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
    Tensor criticState(stateDim + actionDim, 1);
    Tensor::concat(0, criticState,
                   trajectory[end].state,
                   trajectory[end].action);
    float r = critic.forward(criticState).max();
    for (int i = end; i >= 0; i--) {
        r = trajectory[i].reward + gamma * r;
        trajectory[i].reward = r;
    }
    for (int t = end; t >= 0; t--) {
        int k = trajectory[t].action.argmax();
        Tensor::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        /* advangtage */
        Tensor v = critic.forward(criticState);
        float adv = trajectory[t].reward - v[k];
        /* critic */
        Tensor r = v;
        if (t == end) {
            r[k] = trajectory[t].reward;
        } else {
            Tensor::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            Tensor &v1 = critic.forward(criticState);
            r[k] = trajectory[t].reward + 0.99*v1[k];
        }
        critic.backward(Loss::MSE(v, r));
        critic.gradient(criticState, r);
        /* temperture parameter */
        Tensor& q = trajectory[t].action;
        alpha.g[k] += (-q[k]*std::log(q[k] + 1e-8) - entropy0)*alpha[k];
        /* actor */
        Tensor p = actorP.forward(trajectory[t].state);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]) + 1e-9);
        ratio = std::min(ratio, RL::clip(ratio, 1 - epsilon, 1 + epsilon));
        q[k] *= ratio * adv;
        actorP.backward(Loss::CrossEntropy(p, q));
        actorP.gradient(trajectory[t].state, q);
    }
    float decay = annealing.step();
    actorP.RMSProp(learningRate, 0.9, decay);
    critic.RMSProp(1e-3, 0.9, decay);
    alpha.RMSProp(1e-5, 0.9, 0);
#if 1
    std::cout<<"alpha:";
    alpha.val.printValue();
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
