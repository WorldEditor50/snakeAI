#ifndef DRPG_H
#define DRPG_H
#include "net.hpp"
#include "lstm.h"
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
    void reset();
    void reinforce(std::vector<Step>& x, float learningRate);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    Tensor h;
    Tensor c;
    std::shared_ptr<LSTM> lstm;
    Net policyNet;
};
}
#endif // DRPG_H
