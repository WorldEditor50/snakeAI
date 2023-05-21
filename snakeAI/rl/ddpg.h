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
    void perceive(const Mat& state,
                  const Mat& action,
                  const Mat& nextState,
                  float reward,
                  bool done);
    void setSA(const Mat& state, const Mat& action);
    Mat &noiseAction(const Mat &state);
    int randomAction();
    Mat& eGreedyAction(const Mat &state);
    Mat& output() {return actorP.output();}
    int action(const Mat& state);
    void experienceReplay(Transition& x);
    void learn(OptType optType  = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 64,
               float actorLearningRate = 0.0001,
               float criticLearningRate = 0.001);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float beta;
    float exploringRate;
    int learningSteps;
    Mat sa;
    BPNN actorP;
    BPNN actorQ;
    BPNN criticP;
    BPNN criticQ;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
