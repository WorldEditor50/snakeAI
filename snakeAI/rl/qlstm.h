#ifndef QLSTM_H
#define QLSTM_H
#include <deque>
#include "lstm.h"
#include "net.hpp"
#include "rl_basic.h"

namespace RL {

class QLSTM
{
public:
    QLSTM(){}
    explicit QLSTM(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    ~QLSTM(){}
    void perceive(Tensor& state,
                  Tensor& action,
                  Tensor& nextState,
                  float reward,
                  bool done);
    Tensor& eGreedyAction(const Tensor& state);
    Tensor& noiseAction(const Tensor &state);
    Tensor &action(const Tensor &state);
    void reset();
    void experienceReplay(const Transition& x);
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
    Tensor h;
    Tensor c;
    std::shared_ptr<LSTM> lstm;
    Net QMainNet;
    Net QTargetNet;
    std::deque<Transition> memories;
};

}

#endif // QLSTM_H
