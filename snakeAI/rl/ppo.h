#ifndef PPO_H
#define PPO_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "bpnn.h"
#include "lstmnet.h"
#include "rl_basic.h"
namespace RL {

class PPO
{
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim);
    ~PPO(){}
    Vec &eGreedyAction(const Vec &state);
    Vec &action(const Vec &state);
    Vec& output(){return actorP.output();}
    void learnWithKLpenalty(double learningRate, std::vector<Transition>& trajectory);
    void learnWithClipObjective(double learningRate, std::vector<Transition>& x);
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);
protected:
    int stateDim;
    int actionDim;
    double gamma;
    double beta;
    double delta;
    double epsilon;
    double exploringRate;
    int learningSteps;
    LstmNet actorP;
    LstmNet actorQ;
    LstmNet critic;
};
}
#endif // PPO_H
