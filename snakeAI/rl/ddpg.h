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
    explicit DDPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim);
    void perceive(Vec& state,
                  Vec& action,
                  Vec& nextState,
                  double reward,
                  bool done);
    void setSA(Vec& state, Vec& action);
    int noiseAction(Vec& state);
    int randomAction();
    Vec& greedyAction(Vec& state);
    int action(Vec& state);
    void experienceReplay(Transition& x);
    void learn(OptType optType  = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 64,
               double actorLearningRate = 0.0001,
               double criticLearningRate = 0.001);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
public:
    std::size_t stateDim;
    std::size_t actionDim;
    double gamma;
    double alpha;
    double beta;
    int maxMemorySize;
    int batchSize;
    double exploringRate;
    int learningSteps;
    int replaceTargetIter;
    Vec sa;
    BPNN actorP;
    BPNN actorQ;
    BPNN criticMainNet;
    BPNN criticTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
