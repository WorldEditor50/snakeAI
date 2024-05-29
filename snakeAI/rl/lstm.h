#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <memory>
#include "util.h"
#include "activate.h"
#include "optimize.h"
#include "loss.h"

namespace RL {

class LSTMParam
{
public:
    /* input gate */
    Mat wi;
    Mat ui;
    Mat bi;
    /* generate */
    Mat wg;
    Mat ug;
    Mat bg;
    /* forget gate */
    Mat wf;
    Mat uf;
    Mat bf;
    /* output gate */
    Mat wo;
    Mat uo;
    Mat bo;
    /* predict */
    Mat w;
    Mat b;
public:
    LSTMParam(){}
    LSTMParam(const LSTMParam &r)
        :wi(r.wi), ui(r.ui), bi(r.bi),
         wg(r.wg), ug(r.ug), bg(r.bg),
         wf(r.wf), uf(r.uf), bf(r.bf),
         wo(r.wo), uo(r.uo), bo(r.bo),
         w(r.w), b(r.b){}
    LSTMParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        wi = Mat(hiddenDim, inputDim);
        wg = Mat(hiddenDim, inputDim);
        wf = Mat(hiddenDim, inputDim);
        wo = Mat(hiddenDim, inputDim);

        ui = Mat(hiddenDim, hiddenDim);
        ug = Mat(hiddenDim, hiddenDim);
        uf = Mat(hiddenDim, hiddenDim);
        uo = Mat(hiddenDim, hiddenDim);

        bi = Mat(hiddenDim, 1);
        bg = Mat(hiddenDim, 1);
        bf = Mat(hiddenDim, 1);
        bo = Mat(hiddenDim, 1);

        w = Mat(outputDim, hiddenDim);
        b = Mat(outputDim, 1);
    }
    void zero()
    {
        std::vector<Mat*> weights = {&wi, &wg, &wf, &wo,
                                     &ui, &ug, &uf, &uo,
                                     &bi, &bg, &bf, &bo,
                                     &w, &b};
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights[i]->zero();
        }
        return;
    }
    void random()
    {
        std::vector<Mat*> weights = {&wi, &wg, &wf, &wo,
                                     &ui, &ug, &uf, &uo,
                                     &bi, &bg, &bf, &bo,
                                     &w, &b};
        for (std::size_t i = 0; i < weights.size(); i++) {
            RL::uniformRand(*weights[i], -1, 1);
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
        Mat i;
        Mat f;
        Mat g;
        Mat o;
        Mat c;
        Mat h;
        Mat y;
    public:
        State(){}
        State(const State &r)
            :i(r.i),f(r.f), g(r.g),
             o(r.o),c(r.c),h(r.h), y(r.y){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            i(Mat(hiddenDim, 1)),f(Mat(hiddenDim, 1)),g(Mat(hiddenDim, 1)),
            o(Mat(hiddenDim, 1)),c(Mat(hiddenDim, 1)),h(Mat(hiddenDim, 1)),
            y(Mat(outputDim, 1)){}
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
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    Mat h;
    Mat c;
    Mat y;
    /* state */
    std::vector<State> states;
protected:
    float alpha_t;
    float beta_t;
    LSTMParam d;
    LSTMParam v;
    LSTMParam s;
public:
    LSTM(){}
    LSTM(const LSTM &r)
        :inputDim(r.inputDim),hiddenDim(r.hiddenDim),outputDim(r.outputDim),
    h(r.h),c(r.c),y(r.y),states(r.states),alpha_t(r.alpha_t),beta_t(r.beta_t),
    d(r.d),v(r.v),s(r.s){}
    LSTM(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    static std::shared_ptr<LSTM> _(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag)
    {
        return std::make_shared<LSTM>(inputDim_, hiddenDim_, outputDim_, trainFlag);
    }
    void reset();
    Mat &output(){return y;}
    /* forward */
    State feedForward(const Mat &x, const Mat &_h, const Mat &_c);
    void forward(const std::vector<Mat> &sequence);
    Mat &forward(const Mat &x);
    /* backward */
    void backwardAtTime(int t, const Mat &x, const Mat &E, State &delta_);
    void backward(const std::vector<Mat> &x, const std::vector<Mat> &E);
    /* seq2seq */
    void gradient(const std::vector<Mat> &x, const std::vector<Mat> &yt);
    /* seq2Mat */
    void gradient(const std::vector<Mat> &x, const Mat &yt);
    /* optimize */
    void SGD(float learningRate);
    void RMSProp(float learningRate, float rho = 0.9, float decay = 0.01);
    void Adam(float learningRate, float alpha = 0.9, float beta = 0.99, float decay = 0.01);
    void clamp(float c0, float cn);
    /* parameter */
    void copyTo(LSTM &dst);
    void softUpdateTo(LSTM &dst, float rho);
    static void test();
};

}
#endif // LSTM_H
