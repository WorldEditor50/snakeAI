#ifndef DQNN_H
#define DQNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include "net.hpp"
#include "rl_basic.h"

namespace RL {

class DQN
{
public:
    DQN(){}
    explicit DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    Tensor& eGreedyAction(const Tensor& state);
    Tensor& noiseAction(const Tensor& state);
    Tensor& action(const Tensor &state);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    int learningSteps;
    Net QMainNet;
    Net QTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DQNN_H
