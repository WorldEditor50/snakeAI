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
    void perceive(Vec& state,
                  Vec& action,
                  Vec& nextState,
                  double reward,
                  bool done);
    Vec& eGreedyAction(Vec& state);
    Vec& output();
    Vec &action(const Vec &state);
    void reset();
    void experienceReplay(const Transition& x, std::vector<Vec> &y);
    void learn(std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               double learningRate = 0.001);
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    double gamma;
    double exploringRate;
    int learningSteps;
    LstmNet QMainNet;
    LstmNet QTargetNet;
    std::deque<Transition> memories;
};

}

#endif // QLSTM_H
