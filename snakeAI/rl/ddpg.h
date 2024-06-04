#ifndef DDPG_H
#define DDPG_H
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

/* this is not a real DDPG,
 *  DDPG may not work in discrete Action space */
class DDPG
{
public:
    DDPG(){}
    ~DDPG(){}
    explicit DDPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    void perceive(const Tensor& state,
                  const Tensor& action,
                  const Tensor& nextState,
                  float reward,
                  bool done);
    Tensor& noiseAction(const Tensor &state);
    Tensor& action(const Tensor& state);
    void experienceReplay(const Transition& x);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 64);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float beta;
    float exploringRate;
    int learningSteps;
    BPNN actorP;
    BPNN actorQ;
    BPNN criticP;
    BPNN criticQ;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
