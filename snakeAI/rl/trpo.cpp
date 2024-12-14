#include "trpo.h"
#include "layer.h"
#include "loss.h"


RL::TRPO::TRPO(int stateDim_, int hiddenDim, int actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99)
{
    lmbda = 0.95;
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

RL::Tensor &RL::TRPO::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actorQ.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

RL::Tensor &RL::TRPO::action(const RL::Tensor &state)
{
    return actorP.forward(state);
}

RL::Tensor RL::TRPO::hessain(const RL::Tensor &state,
                             const RL::Tensor &oldAction,
                             const RL::Tensor &grad)
{
    Tensor newAction = actorP.forward(state);
    Tensor kl(actionDim, 1);
    for (int i = 0; i < actionDim; i++) {
        kl[i] = newAction[i]*std::log(newAction[i]/oldAction[i]);
    }

    Tensor H;


    return H;
}

void RL::TRPO::learn(std::vector<RL::Step> &x, float learningRate)
{
    /* critic */
    std::size_t N = x.size();
    Tensor tdTarget(N, 1);
    for (std::size_t i = 0; i < N; i++) {
        Tensor &v = x[i].action;
        Tensor qTarget = v;
        Tensor &vn = critic.forward(x[i + 1].state);
        int k = v.argmax();
        if (i == x.size() - 1) {
            qTarget[k] = x[i].reward;
        } else {
            qTarget[k] = x[i].reward + gamma*vn[k];
        }
        tdTarget[i] = qTarget[k];
        critic.backward(Loss::MSE(qTarget, v));
        critic.gradient(x[i].state, qTarget);
    }
    critic.RMSProp(1e-3, 0.9, 0);
    /* actor */
    /* td error */
    Tensor delta(N, 1);
    for (std::size_t i = 0; i < x.size(); i++) {
        float v = x[i].action.max();
        delta[i] = tdTarget[i] - v;
    }
    /* advantage */
    Tensor advantages(x.size(), 1);
    float advantage = 0.0;
    for (std::size_t i = 0; i < N; i++) {
        advantage = gamma*lmbda*advantage + delta[i];
        advantages[N - i - 1] = advantage;
    }
    for (std::size_t i = 0; i < N; i++) {
        Tensor &q = x[i].action;
        int k = q.argmax();
        /* surrogate objective */
        Tensor p  = actorP.forward(x[i].state);
        float ratio = std::exp(std::log(p[k]) - std::log(q[k]));
        float r = ratio * advantages[i];

        /* conjugate gradient */

        /* hessian */

        /* line search */

    }
    return;
}
