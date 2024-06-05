#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"

namespace RL {

class DPG
{
public:
    DPG(){}
    explicit DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~DPG(){}
    Tensor &eGreedyAction(const Tensor &state);
    Tensor &noiseAction(const Tensor &state);
    Tensor &gumbelMax(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    void reinforce(std::vector<Step>& x, float learningRate);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    float entropy0;
    GradValue alpha;
    Net policyNet;
};
}
#endif // POLICY_GRADIENT_H
