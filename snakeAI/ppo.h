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

struct Transit {
    std::vector<double> state;
    std::vector<double> action;
    std::vector<double> nextState;
    double reward;
    Transit(){}
    Transit(std::vector<double>& s,
            std::vector<double>& a,
            std::vector<double>& s_,
            double r)
        :state(s),
          action(a),
          nextState(s_),
          reward(r) {}
};

class PPO
{
public:
    PPO(){}
    ~PPO(){}
    void createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    int greedyAction(std::vector<double>& state);
    int action(std::vector<double>& state);
    double KLmean(std::vector<double>& p, std::vector<double>& q);
    double getValue(std::vector<double> &s);
    int maxAction(std::vector<double>& value);
    void learnWithKLpenalty(int optType, double learningRate, std::vector<Transit>& x);
    void learnWithClip(int optType, double learningRate, std::vector<Transit>& x);
    void save(const std::string &actorPara, const std::string &criticPara);
    void load(const std::string &actorPara, const std::string &criticPara);
    int stateDim;
    int actionDim;
    double gamma;
    double beta;
    double delta;
    double epsilon;
    double exploringRate;
    int learningSteps;
    BPNet actorP;
    BPNet actorQ;
    BPNet critic;
};
}
#endif // PPO_H
