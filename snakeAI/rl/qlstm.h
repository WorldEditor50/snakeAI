#ifndef QLSTM_H
#define QLSTM_H
#include <deque>
#include "lstmnet.h"
#include "rl_basic.h"

namespace RL {

class QLSTM
{
public:
    QLSTM(){}
    explicit QLSTM(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~QLSTM(){}
    void perceive(Mat& state,
                  Mat& action,
                  Mat& nextState,
                  float reward,
                  bool done);
    Mat& eGreedyAction(const Mat& state);
    Mat& output();
    Mat &action(const Mat &state);
    void reset();
    void experienceReplay(const Transition& x, std::vector<Mat> &y);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               float learningRate = 0.001);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    int learningSteps;
    LstmNet QMainNet;
    LstmNet QTargetNet;
    std::deque<Transition> memories;
};

}

#endif // QLSTM_H
