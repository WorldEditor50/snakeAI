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
#include "rl_basic.h"
namespace RL {

class PPO
{
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim);
    ~PPO(){}
    void continousSample(const Vec &state, Vec& act);
    Vec &sample(const Vec &state);
    BPNN &action(const Vec &state);
    Vec& output(){return actorP.output();}
    double learnWithKLpenalty(OptType optType, double learningRate, std::vector<Transition>& x);
    double learnWithClipObject(OptType optType, double learningRate, std::vector<Transition>& x);
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
    BPNN actorP;
    BPNN actorQ;
    BPNN critic;
};
}
#endif // PPO_H
