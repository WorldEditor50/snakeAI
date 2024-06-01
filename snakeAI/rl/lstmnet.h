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
    std::vector<Mat> y;
public:
    LstmNet(){}
    template<typename ...TLayer>
    explicit LstmNet(const LSTM &lstm_, TLayer&&...layer)
        :lstm(lstm_),bpnet(layer...){}
    Mat &forward(const Mat &x)
    {
        Mat &out = lstm.forward(x);
        bpnet.forward(out);
        return bpnet.output();
    }
    void forward(const std::vector<RL::Mat> &sequence)
    {
        lstm.reset();
        for (auto &x : sequence) {
            LSTM::State s = lstm.feedForward(x, lstm.h, lstm.c);
            lstm.h = s.h;
            lstm.c = s.c;
            lstm.states.push_back(s);
            bpnet.forward(s.y);
            y.push_back(bpnet.output());
        }
        return;
    }
    void backward(const std::vector<RL::Mat> &x, const std::vector<RL::Mat> &yt, BPNN::FnLoss Loss)
    {
        std::size_t outputDim = bpnet.output().size();
        RL::Mat E(lstm.outputDim, 1);
        LSTM::State delta_(lstm.hiddenDim, lstm.outputDim);
        for (int t = yt.size() - 1; t >= 0; t--) {
            /* loss */
            Mat loss = Loss(y[t], yt[t]);
            /* backward */
            bpnet.backward(loss, E);
            /* gradient */
            bpnet.gradient(lstm.states[t].y, yt[t]);
            lstm.backwardAtTime(t, x[t], E, delta_);
            E.zero();
        }
        y.clear();
        lstm.states.clear();
        return;
    }

    void copyTo(LstmNet &dst)
    {
        bpnet.copyTo(dst.bpnet);
        lstm.copyTo(dst.lstm);
        return;
    }
    void softUpdateTo(LstmNet &dst, float rho)
    {
        bpnet.softUpdateTo(dst.bpnet, rho);
        lstm.softUpdateTo(dst.lstm, rho);
        return;
    }
    Mat& output(){return bpnet.output();}
    void reset() {lstm.reset();}
    void optimize(float learningRate, float decay)
    {
        lstm.RMSProp(learningRate, 0.9, decay);
        bpnet.RMSProp(0.9, learningRate, decay);
        return;
    }
    void clamp(float c0, float cn)
    {
        bpnet.clamp(c0, cn);
        lstm.clamp(c0, cn);
        return;
    }
};


class LstmStack
{
public:
    std::vector<std::shared_ptr<LSTM> > lstms;
    BPNN bpnet;
    std::vector<Mat> y;
public:
    template <typename ...TLstm>
    LstmStack(TLstm&& ...lstms_):lstms(lstms_...){}
    Mat &forward(const Mat &x)
    {
        LSTM::State s = lstms[0]->feedForward(x, lstms[0]->h, lstms[0]->c);
        lstms[0]->h = s.h;
        lstms[0]->c = s.c;
        lstms[0]->states.push_back(s);
        for (std::size_t i = 1; i < lstms.size(); i++) {
            LSTM::State s = lstms[i]->feedForward(lstms[i - 1]->output(), lstms[i]->h, lstms[i]->c);
            lstms[i]->h = s.h;
            lstms[i]->c = s.c;
            lstms[i]->states.push_back(s);
        }
        Mat out = lstms.back()->output();
        y.push_back(out);
        bpnet.forward(out);
        return bpnet.output();
    }
    void forward(const std::vector<RL::Mat> &sequence)
    {
        for (auto &lstm : lstms) {
            lstm->reset();
        }
        for (auto &x : sequence) {
            LSTM::State s = lstms[0]->feedForward(x, lstms[0]->h, lstms[0]->c);
            lstms[0]->h = s.h;
            lstms[0]->c = s.c;
            lstms[0]->states.push_back(s);
            for (std::size_t i = 1; i < lstms.size(); i++) {
                LSTM::State s = lstms[i]->feedForward(lstms[i - 1]->output(), lstms[i]->h, lstms[i]->c);
                lstms[i]->h = s.h;
                lstms[i]->c = s.c;
                lstms[i]->states.push_back(s);
            }
            y.push_back(lstms.back()->output());
        }
        bpnet.forward(lstms.back()->output());
        return;
    }
    void backward(const std::vector<RL::Mat> &x, const std::vector<RL::Mat> &yt, BPNN::FnLoss Loss)
    {
        std::size_t outputDim = bpnet.output().size();
        auto lstm = lstms.back();
        std::vector<RL::Mat> E(x.size(), Mat(lstm->outputDim, 1));
        std::vector<LSTM::State> deltas(lstms.size(), LSTM::State(lstm->hiddenDim, lstm->outputDim));
        for (std::size_t t = 0; t < yt.size(); t++) {
            /* loss */
            Mat loss = Loss(y[t], yt[t]);
            /* backward */
            bpnet.backward(loss, E[t]);
            /* gradient */
            bpnet.gradient(lstm->states[t].y, yt[t]);
            for (int i = lstms.size() - 1; i >= 0; i--) {
                auto xi = lstms[i]->y;
                lstms[i]->backwardAtTime(t, xi, E[t], deltas[i]);
            }
        }
        y.clear();
        return;
    }
    void copyTo(LstmStack &dst)
    {
        for (std::size_t i = 0; i < lstms.size(); i++) {
            lstms[i]->copyTo(*dst.lstms[i]);
        }
        bpnet.copyTo(dst.bpnet);
        return;
    }
    void softUpdateTo(LstmStack &dst, float rho)
    {
        for (std::size_t i = 0; i < lstms.size(); i++) {
            lstms[i]->softUpdateTo(*dst.lstms[i], rho);
        }
        bpnet.softUpdateTo(dst.bpnet, rho);
        return;
    }
    Mat& output(){return lstms.back()->output();}
    void reset()
    {
        for (auto &lstm : lstms) {
            lstm->reset();
        }
        return;
    }
    void optimize(float learningRate)
    {
        for (auto &lstm : lstms) {
            lstm->RMSProp(learningRate, 0.9);
        }
        bpnet.RMSProp(0.9, learningRate);
        return;
    }
};
}
#endif // LSTMNET_H
