#ifndef DQNN_H
#define DQNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <thread>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "bpnn.h"
#include "rl_basic.h"

namespace RL {

class DQN
{
public:
    DQN(){}
    explicit DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~DQN(){}
    void perceive(const Mat& state,
                  const Mat& action,
                  const Mat& nextState,
                  float reward,
                  bool done);
    Mat& eGreedyAction(const Mat& state);
    Mat& output();
    int action(const Mat &state);
    void annealing(float r);
    void experienceReplay(const Transition& x);
    void learn(OptType optType = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
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
    float totalReward;
    int learningSteps;
    BPNN QMainNet;
    BPNN QTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DQNN_H
