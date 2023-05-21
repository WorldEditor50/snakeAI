#ifndef GRU_H
#define GRU_H
#include <iostream>
#include "util.h"
#include "activate.h"
#include "optimizer.h"
#include "loss.h"

namespace RL {

class GRUParam
{
public:
    /* reset gate */
    Mat Wr;
    Mat Ur;
    Mat Br;
    /* update gate */
    Mat Wz;
    Mat Uz;
    Mat Bz;
    /* h */
    Mat Wg;
    Mat Ug;
    Mat Bg;
    /* predict */
    Mat W;
    Mat B;
public:
    GRUParam(){}
    GRUParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wr = Mat(hiddenDim, inputDim);
        Wz = Mat(hiddenDim, inputDim);
        Wg = Mat(hiddenDim, inputDim);
        Ur = Mat(hiddenDim, hiddenDim);
        Uz = Mat(hiddenDim, hiddenDim);
        Ug = Mat(hiddenDim, hiddenDim);
        Br = Mat(hiddenDim, 1);
        Bz = Mat(hiddenDim, 1);
        Bg = Mat(hiddenDim, 1);
        W = Mat(outputDim, hiddenDim);
        B = Mat(outputDim, 1);
    }

    void zero()
    {
        std::vector<Mat*> weights = {&Wr, &Wz, &Wg,
                                     &Ur, &Uz, &Ug,
                                     &Br, &Bz, &Bg,
                                     &W, &B};
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights[i]->zero();
        }
        return;
    }
    void random()
    {
        std::vector<Mat*> weights = {&Wr, &Wz, &Wg,
                                     &Ur, &Uz, &Ug,
                                     &Br, &Bz, &Bg,
                                     &W, &B};
        for (std::size_t i = 0; i < weights.size(); i++) {
            uniformRand(*weights[i], -1, 1);
        }
        return;
    }
};

class GRU : public GRUParam
{
public:
    class State
    {
    public:
        Mat r;
        Mat z;
        Mat g;
        Mat h;
        Mat y;
    public:
        State(){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            r(Mat(hiddenDim, 1)), z(Mat(hiddenDim, 1)),
            g(Mat(hiddenDim, 1)), h(Mat(hiddenDim, 1)), y(Mat(outputDim, 1)){}
        void zero()
        {
            for (std::size_t k = 0; k < r.size(); k++) {
                r[k] = 0; z[k] = 0; g[k] = 0; h[k] = 0;
            }
            for (std::size_t k = 0; k < y.size(); k++) {
                y[k] = 0;
            }
            return;
        }
    };

protected:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    Mat h;
    /* optimize */
    GRUParam d;
    GRUParam v;
    GRUParam s;
    float alpha_t;
    float beta_t;
    /* state */
    std::vector<State> states;
public:
    GRU(){}
    GRU(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    void clear();
    State feedForward(const Mat &x, const Mat &_h);
    void forward(const std::vector<Mat> &sequence);
    Mat forward(const Mat &x);
    void backward(const std::vector<Mat> &x, const std::vector<Mat> &E);
    void gradient(const std::vector<Mat> &x, const std::vector<Mat> &yt);
    void gradient(const std::vector<Mat> &x, const Mat &yt);
    void SGD(float learningRate = 0.001);
    void RMSProp(float learningRate = 0.001, float rho = 0.9);
    void Adam(float learningRate = 0.001, float alpha = 0.9, float beta = 0.99);
    static void test();
};

}
#endif // GRU_H
