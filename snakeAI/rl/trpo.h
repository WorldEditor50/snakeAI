#ifndef TRPO_H
#define TRPO_H
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

class TRPO
{
public:
    TRPO(){}
    explicit TRPO(int stateDim, int hiddenDim, int actionDim);
    RL::Tensor &eGreedyAction(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    void learn(std::vector<Step>& x, float learningRate);
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);
private:
    int stateDim;
    int actionDim;
    float gamma;
    float maxKL;
    float exploringRate;
    int learningSteps;
    Net actorP;      /* policy network (training) */
    Net actorQ;      /* exploration network (synced from actorP) */
    Net critic;      /* V(s) state-value function with Linear output */

    /* TRPO inner helpers */
    int totalParams();
    Tensor flatParams();
    void setFlatParams(const Tensor &p);
    Tensor flatGrad();
    void zeroGrad();
};

}
#endif // TRPO_H
