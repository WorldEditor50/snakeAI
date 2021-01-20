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
#include "mlp.h"
#include "dpg.h"
namespace ML {

struct Transit {
    Vec state;
    Vec action;
    Vec nextState;
    double reward;
    Transit(){}
    Transit(Vec& s,
            Vec& a,
            Vec& s_,
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
    explicit PPO(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    ~PPO(){}
    int greedyAction(Vec& state);
    int action(Vec& state);
    double KLmean(Vec& p, Vec& q);
    double getValue(Vec &s);
    int maxAction(Vec& value);
    double clip(double x, double sup, double inf);
    void learnWithKLpenalty(OptType optType, double learningRate, std::vector<Transit>& x);
    void learnWithClipObject(OptType optType, double learningRate, std::vector<Transit>& x);
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
    MLP actorP;
    MLP actorQ;
    MLP critic;
};
}
#endif // PPO_H
