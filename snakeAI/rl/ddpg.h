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
    void perceive(const Vec& state,
                  const Vec& action,
                  const Vec& nextState,
                  double reward,
                  bool done);
    void setSA(const Vec& state, const Vec& action);
    int noiseAction(const Vec &state);
    int randomAction();
    Vec& sample(const Vec &state);
    Vec& output() {return actorP.output();}
    int action(const Vec& state);
    void experienceReplay(Transition& x);
    void learn(OptType optType  = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 64,
               double actorLearningRate = 0.0001,
               double criticLearningRate = 0.001);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    double gamma;
    double beta;
    double exploringRate;
    int learningSteps;
    Vec sa;
    BPNN actorP;
    BPNN actorQ;
    BPNN criticP;
    BPNN criticQ;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
