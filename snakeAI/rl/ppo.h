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
#include "net.hpp"
#include "rl_basic.h"
#include "parameter.hpp"
#include "annealing.hpp"

namespace RL {

class PPO
{
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim);
    Tensor &eGreedyAction(const Tensor &state);
    Tensor &noiseAction(const Tensor &state);
    Tensor &gumbelMax(const Tensor &state);
    Tensor &action(const Tensor &state);
    void learnWithKLpenalty(std::vector<Step>& trajectory, float learningRate);
    void learnWithClipObjective(std::vector<Step>& x, float learningRate);
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);
protected:
    int stateDim;
    int actionDim;
    float gamma;
    float beta;
    float delta;
    float epsilon;
    float exploringRate;
    int learningSteps;
    float entropy0;
    GradValue alpha;
    ExpAnnealing annealing;
    Net actorP;
    Net actorQ;
    Net critic;
};
}
#endif // PPO_H
