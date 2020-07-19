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
#include "dqn.h"
namespace ML {

/* this is not a real DDPG,
 *  DDPG may not work in discrete Action space */
class DDPG
{
public:
    DDPG(){}
    ~DDPG(){}
    void CreateNet(int stateDim,
                   int hiddenDim,
                   int hiddenLayerNum,
                   int actionDim,
                   int maxMemorySize = 4096,
                   int replaceTargetIter = 256,
                   int batchSize = 64);
    void Perceive(std::vector<double>& state,
                  double action,
                  std::vector<double>& nextState,
                  double reward,
                  bool done);
    void SetSA(std::vector<double>& state, std::vector<double>& action);
    void Forget();
    int NoiseAction(std::vector<double>& state);
    int RandomAction();
    int GreedyAction(std::vector<double>& state);
    int Action(std::vector<double>& state);
    int MaxQ(std::vector<double>& q_value);
    void ExperienceReplay(Transition& x);
    void Learn(int optType = OPT_RMSPROP,
               double actorLearningRate = 0.01,
               double criticLearningRate = 0.01);
    void Save(const std::string& actorPara, const std::string& criticPara);
    void Load(const std::string& actorPara, const std::string& criticPara);
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
    BPNet actorMainNet;
    BPNet actorTargetNet;
    BPNet criticMainNet;
    BPNet criticTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
