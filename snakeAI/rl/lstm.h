#ifndef LSTM_H
#define LSTM_H

#include <iostream>
#include "rl_basic.h"

namespace RL {

class LSTMParam
{
public:
    /* input gate */
    Mat Wi;
    Mat Ui;
    Vec Bi;
    /* generate */
    Mat Wg;
    Mat Ug;
    Vec Bg;
    /* forget gate */
    Mat Wf;
    Mat Uf;
    Vec Bf;
    /* output gate */
    Mat Wo;
    Mat Uo;
    Vec Bo;
    /* predict */
    Mat W;
    Vec B;
public:
    LSTMParam(){}
    LSTMParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wi = Mat(hiddenDim, Vec(inputDim, 0));
        Wg = Mat(hiddenDim, Vec(inputDim, 0));
        Wf = Mat(hiddenDim, Vec(inputDim, 0));
        Wo = Mat(hiddenDim, Vec(inputDim, 0));

        Ui = Mat(hiddenDim, Vec(hiddenDim, 0));
        Ug = Mat(hiddenDim, Vec(hiddenDim, 0));
        Uf = Mat(hiddenDim, Vec(hiddenDim, 0));
        Uo = Mat(hiddenDim, Vec(hiddenDim, 0));

        Bi = Vec(hiddenDim, 0);
        Bg = Vec(hiddenDim, 0); 
        Bf = Vec(hiddenDim, 0);
        Bo = Vec(hiddenDim, 0);

        W = Mat(outputDim, Vec(hiddenDim, 0));
        B = Vec(outputDim, 0);
    }
    void zero()
    {
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = 0; Wg[i][j] = 0; Wf[i][j] = 0; Wo[i][j] = 0;
            }
        }
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                Ui[i][j] = 0; Ug[i][j] = 0; Uf[i][j] = 0; Uo[i][j] = 0;
            }
            Bi[i] = 0; Bg[i] = 0; Bf[i] = 0; Bo[i] = 0;
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = 0;
            }
            B[i] = 0;
        }
        return;
    }
    void random()
    {
        std::uniform_real_distribution<double> uniform(-1, 1);
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = uniform(Rand::engine);
                Wg[i][j] = uniform(Rand::engine);
                Wf[i][j] = uniform(Rand::engine);
                Wo[i][j] = uniform(Rand::engine);
            }
        }
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                Ui[i][j] = uniform(Rand::engine);
                Ug[i][j] = uniform(Rand::engine);
                Uf[i][j] = uniform(Rand::engine);
                Uo[i][j] = uniform(Rand::engine);
            }
            Bi[i] = uniform(Rand::engine);
            Bg[i] = uniform(Rand::engine);
            Bf[i] = uniform(Rand::engine);
            Bo[i] = uniform(Rand::engine);
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = uniform(Rand::engine);
            }
            B[i] = uniform(Rand::engine);
        }
        return;
    }
};

class LSTM : public LSTMParam
{
public:
    class State
    {
    public:
        Vec i;
        Vec f;
        Vec g;
        Vec o;
        Vec c;
        Vec h;
        Vec y;
    public:
        State(){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            i(Vec(hiddenDim, 0)),f(Vec(hiddenDim, 0)),g(Vec(hiddenDim, 0)),
            o(Vec(hiddenDim, 0)),c(Vec(hiddenDim, 0)),h(Vec(hiddenDim, 0)),
            y(Vec(outputDim, 0)){}
        void zero()
        {
            for (std::size_t k = 0; k < i.size(); k++) {
                i[k] = 0; f[k] = 0; g[k] = 0;
                o[k] = 0; c[k] = 0; h[k] = 0;
            }
            for (std::size_t k = 0; k < y.size(); k++) {
                y[k] = 0;
            }
            return;
        }
    };
public:
    Vec h;
    Vec c;
    Vec y;
    /* state */
    std::vector<State> states;
protected:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    LSTMParam d;
    LSTMParam v;
    LSTMParam s;
    double alpha_t;
    double beta_t;
public:
    LSTM(){}
    LSTM(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    void clear();
    /* forward */
    State feedForward(const Vec &x, const Vec &_h, const Vec &_c);
    void forward(const std::vector<Vec> &sequence);
    Vec &forward(const Vec &x);
    /* backward */
    void backward(const std::vector<Vec> &x, const std::vector<Vec> &E);
    /* seq2seq */
    void gradient(const std::vector<Vec> &x, const std::vector<Vec> &yt);
    /* seq2vec */
    void gradient(const std::vector<Vec> &x, const Vec &yt);
    /* optimize */
    void SGD(double learningRate);
    void RMSProp(double learningRate, double rho = 0.9);
    void Adam(double learningRate, double alpha = 0.9, double beta = 0.99);
    /* parameter */
    void copyTo(LSTM &dst);
    void softUpdateTo(LSTM &dst, double rho);
    static void test();
};

}
#endif // LSTM_H
