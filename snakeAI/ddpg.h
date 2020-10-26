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
#include "mlp.h"
#include "dqn.h"
namespace ML {

/* this is not a real DDPG,
 *  DDPG may not work in discrete Action space */
class DDPG
{
public:
    DDPG(){}
    ~DDPG(){}
    explicit DDPG(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    void perceive(std::vector<double>& state,
                  std::vector<double>& action,
                  std::vector<double>& nextState,
                  double reward,
                  bool done);
    void setSA(std::vector<double>& state, std::vector<double>& action);
    void forget();
    int noiseAction(std::vector<double>& state);
    int randomAction();
    std::vector<double>& greedyAction(std::vector<double>& state);
    int action(std::vector<double>& state);
    int maxQ(std::vector<double>& q_value);
    void experienceReplay(Transition& x);
    void learn(int optType  = OPT_RMSPROP,
               int maxMemorySize = 4096,
               int replaceTargetIter = 256,
               int batchSize = 64,
               double actorLearningRate = 0.0001,
               double criticLearningRate = 0.001);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
    int stateDim;
    int actionDim;
    double gamma;
    double alpha;
    double beta;
    int maxMemorySize;
    int batchSize;
    double exploringRate;
    int learningSteps;
    int replaceTargetIter;
    std::vector<double> sa;
    MLP actorP;
    MLP actorQ;
    MLP criticMainNet;
    MLP criticTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
