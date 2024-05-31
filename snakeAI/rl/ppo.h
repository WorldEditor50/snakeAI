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
#include  "parameter.hpp"

namespace RL {

class PPO
{
public:
    PPO(){}
    explicit PPO(int stateDim, int hiddenDim, int actionDim);
    ~PPO(){}
    Mat &eGreedyAction(const Mat &state);
    RL::Mat &noiseAction(const RL::Mat &state);
    RL::Mat &gumbelMax(const RL::Mat &state);
    Mat &action(const Mat &state);
    Mat& output(){return actorP.output();}
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
    BPNN actorP;
    BPNN actorQ;
    BPNN critic;
};
}
#endif // PPO_H
