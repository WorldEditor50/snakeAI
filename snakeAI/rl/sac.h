#ifndef SAC_H
#define SAC_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"

namespace RL {


class SAC
{
public:
    SAC(){}
    explicit SAC(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    void perceive(const std::vector<Transition>& x);
    Tensor& eGreedyAction(const Tensor& state);
    Tensor& gumbelMax(const Tensor &state);
    Tensor& action(const Tensor &state);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
    void save();
    void load();
protected:
    int stateDim;
    int actionDim;
    float gamma;
    float entropy0;
    float exploringRate;
    int learningSteps;
    std::deque<Transition> memories;
    GradValue alpha;
    Net actor;
    Net critic1Net;
    Net critic1TargetNet;
    Net critic2Net;
    Net critic2TargetNet;
};

}
#endif // SAC_H
