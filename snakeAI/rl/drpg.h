#ifndef DRPG_H
#define DRPG_H
#include "lstmnet.h"
#include "rl_basic.h"

namespace RL {

class DRPG
{
public:
    DRPG(){}
    explicit DRPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~DRPG(){}
    Vec &eGreedyAction(const Vec &state);
    Vec &output(){return policyNet.output();}
    Vec &action(const Vec &state);
    void reinforce(const std::vector<Vec> &x,
                   std::vector<Vec> &y,
                   std::vector<double>& reward,
                   double learningRate);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    double gamma;
    double exploringRate;
    double learningRate;
    LstmNet policyNet;
};
}
#endif // DRPG_H
