#ifndef DRPG_H
#define DRPG_H
#include "lstmnet.h"
#include "rl_basic.h"

namespace RL {

class DRPG
{
public:
    DRPG(){}
    explicit DRPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~DRPG(){}
    Tensor &eGreedyAction(const Tensor &state);
    RL::Tensor &noiseAction(const RL::Tensor &state);
    RL::Tensor &gumbelMax(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    void reinforce(const std::vector<Tensor> &x,
                   std::vector<Tensor> &y,
                   std::vector<float>& reward,
                   float learningRate);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    LstmNet policyNet;
};
}
#endif // DRPG_H
