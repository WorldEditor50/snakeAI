#ifndef BCQ_H
#define BCQ_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include "net.hpp"
#include "rl_basic.h"
#include "vae.hpp"

namespace RL {

class BCQ
{
public:
    struct Step {
        Tensor state;
        Tensor action;
    };
public:
    BCQ();
    explicit BCQ(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    Tensor& action(const Tensor &state);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    std::size_t featureDim;
    float gamma;
    float exploringRate;
    int learningSteps;
    VAE stateGenerator;
    VAE actionGenerator;
    Net qNet1;
    Net qNet2;
    std::deque<Transition> memories;
};

}
#endif // BCQ_H
