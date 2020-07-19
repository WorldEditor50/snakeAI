#ifndef PPO_H
#define PPO_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "bpnn.h"
#include "dpg.h"
namespace ML {

class PPO
{
public:
    PPO(){}
    ~PPO(){}
    void CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                   int maxMemorySize = 1024, int replaceTargetIter = 256, int batchSize = 64);
    int GreedyAction(std::vector<double>& state);
    int Action(std::vector<double>& state);
    void Perceive(std::vector<Step>& trajectory);
    void ExperienceReplay(std::vector<Step>& trajectory);
    void Learn(int optType = OPT_RMSPROP, double learningRate = 0.001);
    void Save(const std::string &actorPara, const std::string &criticPara);
    void Load(const std::string &actorPara, const std::string &criticPara);
    int stateDim;
    int actionDim;
    double gamma;
    double exploringRate;
    double epsilon;
    double learningRate;
    int maxMemorySize;
    int replaceTargetIter;
    int batchSize;
    int learningStep;
    BPNet actor;
    BPNet actorPrime;
    BPNet critic;
    std::deque<std::vector<Step> > memories;
};
}
#endif // PPO_H
