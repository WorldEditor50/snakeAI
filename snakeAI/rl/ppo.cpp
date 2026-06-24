#include "ppo.h"
#include "layer.h"
#include "loss.h"

RL::PPO::PPO(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99),
      beta(0.5),delta(0.01),epsilon(0.2), exploringRate(1),learningSteps(0)
{
    annealing = ExpAnnealing(0.01, 0.12);
    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    H0 = RL::entropy(0.25);

    actorP = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, true),
                 //TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 //Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, true));

    actorQ = Net(Layer<Tanh>::_(stateDim, hiddenDim, true, false),
                 //TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                 //Layer<Tanh>::_(hiddenDim, hiddenDim, true, false),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, false),
                 Layer<Softmax>::_(hiddenDim, actionDim, true, false));

    critic = Net(Layer<Tanh>::_(stateDim + actionDim, hiddenDim, true, true),
                 //TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 //Layer<Tanh>::_(hiddenDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Relu>::_(hiddenDim, actionDim, true, true));

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

    /* compute discounted returns */
    std::vector<float> returns(trajectory.size(), 0);
    returns[end] = trajectory[end].reward;
    for (int i = end - 1; i >= 0; i--) {
        returns[i] = trajectory[i].reward + gamma * returns[i+1];
    }
    float KLexpect = 0;
    for (int t = end; t >= 0; t--) {
        /* freeze old policy */
        const Tensor q = trajectory[t].action;
        int k = q.argmax();
        /* advantage: A(s,a) = R - V(s,a) */
        Tensor::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        Tensor v = critic.forward(criticState);
        float advantage = returns[t] - v[k];
        /* critic TD target */
        Tensor r = v;
        if (t == end) {
            r[k] = returns[t];
        } else {
            Tensor::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            Tensor &v1 = critic.forward(criticState);
            r[k] = returns[t];
        }
        critic.backward(criticState, Loss::MSE::df(v, r));
        /* temperture parameter */
        float H = RL::entropy(q[k]);
        alpha.g[k] += H0 - H;
        /* actor using old policy as baseline */
        Tensor p = actorP.forward(trajectory[t].state);
        float kl = p[k] * std::log(p[k]/(q[k] + 1e-9));
        float ratio = std::exp(std::log(p[k] + 1e-9) - std::log(q[k] + 1e-9));
        Tensor dLoss(actionDim, 1);
        dLoss[k] = p[k] - ratio*advantage + beta*kl;
        actorP.backward(trajectory[t].state, dLoss);
        KLexpect += kl;
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
    Tensor criticState(stateDim + actionDim, 1);
    /* compute discounted returns */
    int end = trajectory.size() - 1;
    std::vector<float> returns(trajectory.size(), 0);
    returns[end] = trajectory[end].reward;
    for (int i = end - 1; i >= 0; i--) {
        returns[i] = trajectory[i].reward + gamma * returns[i+1];
    }
    for (int t = end; t >= 0; t--) {
        const Tensor q = trajectory[t].action;
        int k = q.argmax();
        Tensor::concat(0, criticState,
                    trajectory[t].state,
                    trajectory[t].action);
        /* advantage */
        Tensor v = critic.forward(criticState);
        float advantage = returns[t] - v[k];
        /* critic */
        Tensor r = v;
        if (t == end) {
            r[k] = returns[t];
        } else {
            Tensor::concat(0, criticState,
                        trajectory[t + 1].state,
                        trajectory[t + 1].action);
            Tensor &v1 = critic.forward(criticState);
            r[k] = returns[t];
        }
        critic.backward(criticState, Loss::MSE::df(v, r));
        /* temperture parameter */
        float H = RL::entropy(q[k]);
        alpha.g[k] += H0 - H;
        /* actor */
        Tensor p = actorP.forward(trajectory[t].state);
        float ratio = std::exp(std::log(p[k] + 1e-8) - std::log(q[k] + 1e-8));
        float surr1 = ratio*advantage;
        float surr2 = RL::clip(ratio, 1 - epsilon, 1 + epsilon)*advantage;
        Tensor dLoss(actionDim, 1);
        dLoss[k] = p[k] - std::min(surr1, surr2);
        actorP.backward(trajectory[t].state, dLoss);
    }
    float decay = annealing.step();
    actorP.RMSProp(learningRate, 0.9, decay);
    critic.RMSProp(1e-3, 0.9, decay);
    alpha.RMSProp(1e-7, 0.9, 0);
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
