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
#include "bpnn.h"
#include "rl_basic.h"
namespace RL {

class DPG
{
public:
    DPG(){}
    explicit DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~DPG(){}
    Mat &eGreedyAction(const Mat &state);
    Mat &output(){return policyNet.output();}
    int action(const Mat &state);
    void reinforce(OptType optType, float learningRate, std::vector<Step>& x);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    BPNN policyNet;
};
}
#endif // POLICY_GRADIENT_H
