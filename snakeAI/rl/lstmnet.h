#ifndef LSTMNET_H
#define LSTMNET_H
#include "lstm.h"
#include "layer.h"
#include "bpnn.h"
namespace RL {
class LstmNet
{
public:
    LSTM lstm;
    BPNN bpnet;
    std::vector<Vec> y;
public:
    LstmNet(){}
    template<typename ...TLayer>
    explicit LstmNet(const LSTM &lstm_, TLayer&&...layer)
        :lstm(lstm_),bpnet(layer...){}
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
        std::size_t outputDim = bpnet.output().size();
        std::vector<RL::Vec> E(x.size(), Vec(lstm.outputDim, 0));
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
        return;
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
}
#endif // LSTMNET_H
