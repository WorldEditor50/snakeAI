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
    void Perceive(std::vector<float>& state,
                  float action,
                  std::vector<float>& nextState,
                  float reward,
                  bool done);
    void SetSA(std::vector<float>& state, std::vector<float>& action);
    void Forget();
    int NoiseAction(std::vector<float>& state);
    int RandomAction();
    int GreedyAction(std::vector<float>& state);
    int Action(std::vector<float>& state);
    int MaxQ(std::vector<float>& q_value);
    void ExperienceReplay(Transition& x);
    void Learn(int optType = OPT_RMSPROP,
               float actorLearningRate = 0.01f,
               float criticLearningRate = 0.01f);
    void Save(const std::string& actorPara, const std::string& criticPara);
    void Load(const std::string& actorPara, const std::string& criticPara);
    int stateDim;
    int actionDim;
    float gamma;
    float alpha;
    float beta;
    int maxMemorySize;
    int batchSize;
    float exploringRate;
    int learningSteps;
    int replaceTargetIter;
    std::vector<float> sa;
    BPNet actorMainNet;
    BPNet actorTargetNet;
    BPNet criticMainNet;
    BPNet criticTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
