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
    static constexpr int max_qnet_num = 2;
    /* Number of candidate actions sampled from VAE prior per state */
    static constexpr int NUM_CANDIDATES = 100;
public:
    BCQ(){}
    explicit BCQ(int stateDim, int hiddenDim, int actionDim);
    /*
     * Standard BCQ action: sample NUM_CANDIDATES candidates from VAE prior,
     * run each through actor + critics, pick the one with highest Q.
     */
    Tensor& action(const Tensor &state);
    Tensor& mixAction(const Tensor &state, const Tensor &ga);
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
protected:
    int stateDim;
    int actionDim;
    int featureDim;
    float gamma;
    int learningSteps;

    /* Internal buffers for action selection (avoids reallocation) */
    Tensor actionOut;
    Tensor bestQOut;

    VAE encoder;
    Net actor;
    Net critics[max_qnet_num];
    Net criticsTarget[max_qnet_num];
    std::deque<Transition> memories;
};

}
#endif // BCQ_H
