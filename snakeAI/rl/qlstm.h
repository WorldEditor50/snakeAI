#ifndef QLSTM_H
#define QLSTM_H
#include <deque>
#include "lstm.h"
#include "layer.h"
#include "bpnn.h"

namespace RL {

class LstmNet
{
public:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    LSTM lstm;
    BPNN bpnet;
    std::vector<Vec> y;
public:
    LstmNet(){}
    LstmNet(std::size_t inputDim_,
         std::size_t hiddenDim_,
         std::size_t outputDim_,
         const BPNN::Layers &layers,
         bool trainFlag)
        :inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_),
          bpnet(layers)
    {
        lstm = LSTM(inputDim_, hiddenDim_, hiddenDim_, trainFlag);
    }
    Vec &forward(const Vec &state)
    {
        bpnet.feedForward(lstm.forward(state));
        return bpnet.output();
    }
    void forward(const std::vector<RL::Vec> &sequence)
    {
        lstm.reset();
        for (auto &x : sequence) {
            LSTM::State s = lstm.feedForward(x, lstm.h, lstm.c);
            lstm.h = s.h;
            lstm.c = s.c;
            lstm.states.push_back(s);
            bpnet.feedForward(s.y);
            y.push_back(bpnet.output());
        }
        return;
    }
    void backward(const std::vector<RL::Vec> &x, const std::vector<RL::Vec> &yt, BPNN::LossFunc Loss)
    {
        std::vector<RL::Vec> E(x.size(), Vec(hiddenDim, 0));
        for (std::size_t t = 0; t < yt.size(); t++) {
            /* loss */
            Vec loss(outputDim, 0);
            Loss(loss, y[t], yt[t]);
            /* backward */
            bpnet.backward(loss, E[t]);
            /* gradient */
            bpnet.gradient(lstm.states[t].y, yt[t]);
        }
        y.clear();
        lstm.backward(x, E);
        lstm.gradient(x, yt);
        return;
    }
    void copyTo(LstmNet &dst)
    {
        bpnet.copyTo(dst.bpnet);
        lstm.copyTo(dst.lstm);
    }
    void softUpdateTo(LstmNet &dst, double rho)
    {
        bpnet.softUpdateTo(dst.bpnet, rho);
        lstm.softUpdateTo(dst.lstm, rho);
        return;
    }
    Vec& output(){return bpnet.output();}
    void reset() {lstm.reset();}
    void optimize(double learningRate)
    {
        lstm.RMSProp(learningRate);
        bpnet.RMSProp(0.9, learningRate);
        return;
    }
};

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
    Vec& sample(Vec& state);
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
