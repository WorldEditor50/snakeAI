#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <memory>
#include "util.h"
#include "activate.h"
#include "optimizer.h"
#include "loss.h"

namespace RL {

class LSTMParam
{
public:
    /* input gate */
    Mat Wi;
    Mat Ui;
    Mat Bi;
    /* generate */
    Mat Wg;
    Mat Ug;
    Mat Bg;
    /* forget gate */
    Mat Wf;
    Mat Uf;
    Mat Bf;
    /* output gate */
    Mat Wo;
    Mat Uo;
    Mat Bo;
    /* predict */
    Mat W;
    Mat B;
public:
    LSTMParam(){}
    LSTMParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wi = Mat(hiddenDim, inputDim);
        Wg = Mat(hiddenDim, inputDim);
        Wf = Mat(hiddenDim, inputDim);
        Wo = Mat(hiddenDim, inputDim);

        Ui = Mat(hiddenDim, hiddenDim);
        Ug = Mat(hiddenDim, hiddenDim);
        Uf = Mat(hiddenDim, hiddenDim);
        Uo = Mat(hiddenDim, hiddenDim);

        Bi = Mat(hiddenDim, 1);
        Bg = Mat(hiddenDim, 1);
        Bf = Mat(hiddenDim, 1);
        Bo = Mat(hiddenDim, 1);

        W = Mat(outputDim, hiddenDim);
        B = Mat(outputDim, 1);
    }
    void zero()
    {
        std::vector<Mat*> weights = {&Wi, &Wg, &Wf, &Wo,
                                     &Ui, &Ug, &Uf, &Uo,
                                     &Bi, &Bg, &Bf, &Bo,
                                     &W, &B};
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights[i]->zero();
        }
        return;
    }
    void random()
    {
        std::vector<Mat*> weights = {&Wi, &Wg, &Wf, &Wo,
                                     &Ui, &Ug, &Uf, &Uo,
                                     &Bi, &Bg, &Bf, &Bo,
                                     &W, &B};
        for (std::size_t i = 0; i < weights.size(); i++) {
            RL::uniformRand(*weights[i], -1, 1);
        }
        return;
    }
};

class LSTM : public LSTMParam
{
public:
    struct Gamma {
        float i;
        float f;
        float g;
        float o;
    };
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
    bool ema;
    float gamma;
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
    void clamp(float c);
    /* parameter */
    void copyTo(LSTM &dst);
    void softUpdateTo(LSTM &dst, float rho);
    static void test();
};

class LstmAttention
{
public:
    std::size_t hiddenDim;
    std::size_t attentionDim;
    Mat w;
    Mat u;
    Mat y;
public:
    explicit LstmAttention(std::size_t hiddenDim_, std::size_t attentionDim_)
        :hiddenDim(hiddenDim_), attentionDim(attentionDim_)
    {
        w = Mat(hiddenDim, attentionDim);
        u = Mat(attentionDim, 1);
        y = Mat(hiddenDim, 1);
        uniformRand(w, -1, 1);
        uniformRand(u, -1, 1);

    }
    void forward(const std::vector<Mat> &o)
    {
        /* yi = softmax(tanh(o*w)*u) ⊙ o*/

        std::size_t sequenceLen = o.size();
        /*
            o1 = tanh(o*w)
            o:[sequence length, hiddenDim]
            w:[hiddenDim, attentionDim]
            o1:[sequence length, attentionDim]
        */
        Mat o1(sequenceLen, attentionDim);
        for (std::size_t i = 0; i < sequenceLen; i++) {
            for (std::size_t j = 0; j < attentionDim; j++) {
                for (std::size_t k = 0; k < hiddenDim; k++) {
                    o1(i, j) += o[i][k]*w(k, j);
                }
            }
        }
        for (std::size_t i = 0; i < sequenceLen; i++) {
            for (std::size_t j = 0; j < attentionDim; j++) {
                o1(i, j) = std::tanh(o1(i, j));
            }
        }
        /*
            o2 = o1*u
            o1:[sequence length, attentionDim]
            u:[attentionDim, 1]
            o2:[sequence length, 1]
        */
        Mat o2(sequenceLen, 1);
        for (std::size_t i = 0; i < sequenceLen; i++) {
            for (std::size_t j = 0; j < attentionDim; j++) {
                o2[i] += o1(i, j)*u[j];
            }
        }
        /*
            alpha = softmax(o2)
            alpha:[sequence length, 1]
        */
        float s = 0;
        for (std::size_t i = 0; i < sequenceLen; i++) {
             s += exp(o2[i]);
        }
        Mat alpha(sequenceLen, 1);
        for (std::size_t i = 0; i < sequenceLen; i++) {
             alpha[i] = exp(o2[i])/s;
        }
        /*
             o3 = o ⊙ alpha
             o:[sequence length, hiddenDim]
             alpha:[sequence length, 1]
             o3:[sequence length, hiddenDim]
        */
        Mat o3(sequenceLen, hiddenDim);
        for (std::size_t i = 0; i < sequenceLen; i++) {
            for (std::size_t j = 0; j < hiddenDim; j++) {
                o3(i, j) = o[i][j]*alpha[i];
            }
        }
        /*
            y = sum(o3)
            o3:[sequence length, hiddenDim]
            y:[hiddenDim, 1]
        */
        for (std::size_t i = 0; i < hiddenDim; i++) {
            for (std::size_t j = 0; j < sequenceLen; j++) {
                y[i] += o3(j, i);
            }
        }
        return;
    }
    void backward(const std::vector<Mat> &o, const std::vector<Mat> &E)
    {
        /*
             yi = softmax(tanh(o*w)*u) ⊙ o
             dyi/dw = dyi/dalpha ⊙ o
             dyi/dalpha = exp(o2)/s * do2/do1
             do2/do1 = u * do1/dw
             do1/dw = (1 - o1^2)*o
             do2/du = tanh(o*w)
        */
    }
};
}
#endif // LSTM_H
