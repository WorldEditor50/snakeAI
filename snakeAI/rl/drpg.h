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
    Mat &eGreedyAction(const Mat &state);
    Mat &output(){return policyNet.output();}
    Mat &action(const Mat &state);
    void reinforce(const std::vector<Mat> &x,
                   std::vector<Mat> &y,
                   std::vector<float>& reward,
                   float learningRate);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    LstmNet policyNet;
};
}
#endif // DRPG_H
