#ifndef QLSTM_H
#define QLSTM_H
#include <deque>
#include "lstm.h"
#include "layer.h"

namespace RL {
class QNet
{
public:
    LSTM lstm;
    Layer<Sigmoid> layer1;
    Layer<Sigmoid> layer2;
    std::vector<Vec> y;
    std::size_t stateDim;
    std::size_t hiddenDim;
    std::size_t actionDim;
public:
    QNet(){}
    QNet(std::size_t stateDim_,
         std::size_t hiddenDim_,
         std::size_t actionDim_,
         bool trainFlag)
        :stateDim(stateDim_), hiddenDim(hiddenDim_), actionDim(actionDim_)
    {
        lstm = LSTM(stateDim_, hiddenDim_, hiddenDim_, trainFlag);
        layer1 = Layer<Sigmoid>(hiddenDim_, hiddenDim_, trainFlag);
        layer2 = Layer<Sigmoid>(hiddenDim_, actionDim_, trainFlag);
    }
    Vec &forward(const Vec &state)
    {
        layer1.feedForward(lstm.forward(state));
        layer2.feedForward(layer1.O);
        return layer2.O;
    }
    void forward(const std::vector<RL::Vec> &sequence)
    {
        lstm.clear();
        for (auto &x : sequence) {
            LSTM::State s = lstm.feedForward(x, lstm.h, lstm.c);
            lstm.h = s.h;
            lstm.c = s.c;
            lstm.states.push_back(s);
            layer1.feedForward(lstm.y);
            layer2.feedForward(layer1.O);
            y.push_back(layer2.O);
        }
        return;
    }
    void backward(const std::vector<RL::Vec> &x, const std::vector<RL::Vec> &yt)
    {
        Vec tmp;
        std::vector<RL::Vec> E(x.size(), Vec(hiddenDim, 0));
        for (std::size_t t = 0; t < yt.size(); t++) {
            /* loss */
            for (std::size_t i = 0; i < actionDim; i++) {
                layer2.E[i] = 2 * (y[t][i] - yt[t][i]);
            }
            layer1.backward(layer2.E, layer2.W);
            for (std::size_t i = 0; i < layer1.W.size(); i++) {
                for (std::size_t j = 0; j < layer1.W[0].size(); j++) {
                    E[t][j] += layer1.W[i][j] * layer1.E[i];
                }
            }
            /* gradient */
            layer2.gradient(layer1.O, tmp);
            layer1.gradient(lstm.y, tmp);
        }
        y.clear();
        lstm.backward(x, E);
        return;
    }
    void copyTo(QNet &dst)
    {
        for (std::size_t i = 0; i < layer2.W.size(); i++) {
            for (std::size_t j = 0; j < layer2.W[0].size(); j++) {
                dst.layer2.W[i][j] = layer2.W[i][j];
            }
            dst.layer2.B[i] = layer2.B[i];
        }
        for (std::size_t i = 0; i < layer1.W.size(); i++) {
            for (std::size_t j = 0; j < layer1.W[0].size(); j++) {
                dst.layer1.W[i][j] = layer1.W[i][j];
            }
            dst.layer1.B[i] = layer1.B[i];
        }
        lstm.copyTo(dst.lstm);
    }
    void softUpdateTo(QNet &dst, double rho)
    {
        for (std::size_t i = 0; i < layer2.W.size(); i++) {
            RL::EMA(dst.layer2.W[i], layer2.W[i], rho);
        }
        RL::EMA(dst.layer2.B, layer2.B, rho);
        for (std::size_t i = 0; i < layer1.W.size(); i++) {
            RL::EMA(dst.layer1.W[i], layer1.W[i], rho);
        }
        RL::EMA(dst.layer1.B, layer1.B, rho);
        lstm.softUpdateTo(dst.lstm, rho);
    }
    Vec& output(){return layer2.O;}
    void clear() {lstm.clear();}
    void optimize(double learningRate)
    {
        lstm.RMSProp(learningRate);
        Optimizer::RMSProp(layer1.d.W, layer1.s.W, layer1.W, learningRate, 0.9);
        Optimizer::RMSProp(layer1.d.B, layer1.s.B, layer1.B, learningRate, 0.9);
        layer1.d.zero();
        Optimizer::RMSProp(layer2.d.W, layer2.s.W, layer2.W, learningRate, 0.9);
        Optimizer::RMSProp(layer2.d.B, layer2.s.B, layer2.B, learningRate, 0.9);
        layer2.d.zero();
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
    void clear();
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
    LSTM QMainNet;
    LSTM QTargetNet;
    std::deque<Transition> memories;
};

}

#endif // QLSTM_H
