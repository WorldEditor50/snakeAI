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
#include "bpnn.h"
#include "rl_basic.h"
#include "parameter.hpp"

namespace RL {


class SAC
{
public:
    SAC(){}
    explicit SAC(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~SAC(){}
    void perceive(const Mat& state,
                  const Mat& action,
                  const Mat& nextState,
                  float reward,
                  bool done);
    Mat& eGreedyAction(const Mat& state);
    Mat& action(const Mat &state);
    void experienceReplay(const Transition& x);
    void learn(OptType optType = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
protected:
    int stateDim;
    int actionDim;
    float gamma;
    float epsilon;
    float exploringRate;
    int learningSteps;
    std::deque<Transition> memories;
    GradValue alpha;
    BPNN actor;
    BPNN critic1Net;
    BPNN critic1TargetNet;
    BPNN critic2Net;
    BPNN critic2TargetNet;

};

}
#endif // SAC_H
