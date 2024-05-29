#include "dpg.h"
RL::DPG::DPG(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.9;
    exploringRate = 1;
    stateDim = stateDim_;
    actionDim = actionDim_;

    alpha = GradValue(actionDim, 1);
    alpha.val.fill(1);
    entropy0 = -0.12*std::log(0.12);//0.25
    policyNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                     LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true),
                     Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                     LayerNorm<Sigmoid, LN::Post>::_(hiddenDim, hiddenDim, true),
                     Softmax::_(hiddenDim, actionDim, true));
}

RL::Mat &RL::DPG::eGreedyAction(const Mat &state)
{
    Mat& out = policyNet.forward(state);
    return eGreedy(out, exploringRate, false);
}

RL::Mat &RL::DPG::noiseAction(const RL::Mat &state)
{
    Mat& out = policyNet.forward(state);
    return noise(out);
}

RL::Mat &RL::DPG::gumbelMax(const RL::Mat &state)
{
    Mat& out = policyNet.forward(state);
    return gumbelSoftmax(out, alpha.val);
}

int RL::DPG::action(const Mat &state)
{
    int a = policyNet.forward(state).argmax();
    policyNet.show();
    return a;
}

void RL::DPG::reinforce(float learningRate, std::vector<Step>& x)
{
    float r = 0;
    Mat discountedReward(x.size(), 1);
    for (int i = x.size() - 1; i >= 0; i--) {
        r = gamma * r + x[i].reward;
        discountedReward[i] = r;
    }
    float u = discountedReward.mean();
    for (std::size_t i = 0; i < x.size(); i++) {
        const Mat &p = x[i].action;   
        int k = x[i].action.argmax();
        alpha.g[k] += -p[k]*std::log(p[k] + 1e-8) - entropy0;
        float ri = discountedReward[i] - u;
        x[i].action[k] *= ri;
        policyNet.gradient(x[i].state, x[i].action, Loss::CrossEntropy);
    }
    alpha.RMSProp(0.9, 1e-4, 0);
    alpha.clamp(0.1, 1.25);
#if 1
    std::cout<<"alpha:";
    for (std::size_t i = 0; i < actionDim; i++) {
        std::cout<<alpha[i]<<" ";
    }
    std::cout<<std::endl;
#endif

    policyNet.optimize(OPT_NORMRMSPROP, learningRate, 0.1);
    policyNet.clamp(-1, 1);
    exploringRate *= 0.9999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    return;
}

void RL::DPG::save(const std::string &fileName)
{
    policyNet.save(fileName);
    return;
}

void RL::DPG::load(const std::string &fileName)
{
    policyNet.load(fileName);
    return;
}
