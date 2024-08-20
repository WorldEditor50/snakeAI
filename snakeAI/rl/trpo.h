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
    RL::Tensor &gumbelMax(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    Tensor hessain(const Tensor &state,
                   const Tensor &oldAction,
                   const Tensor &grad);
    void learn(std::vector<Step>& x, float learningRate);
private:
    int stateDim;
    int actionDim;
    float gamma;
    float lmbda;
    float maxKL;
    int learningSteps;
    float entropy0;
    GradValue alpha;
    ExpAnnealing annealing;
    Net actorP;
    Net actorQ;
    Net critic;
};

}
#endif // TRPO_H
