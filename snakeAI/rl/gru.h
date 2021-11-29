#ifndef GRU_H
#define GRU_H

#include <iostream>
#include "rl_basic.h"

namespace RL {

class GRUParam
{
public:
    /* reset gate */
    Mat Wr;
    Mat Ur;
    Vec Br;
    /* update gate */
    Mat Wz;
    Mat Uz;
    Vec Bz;
    /* h */
    Mat Wg;
    Mat Ug;
    Vec Bg;
    /* predict */
    Mat W;
    Vec B;
public:
    GRUParam(){}
    GRUParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wr = Mat(hiddenDim, Vec(inputDim, 0));
        Wz = Mat(hiddenDim, Vec(inputDim, 0));
        Wg = Mat(hiddenDim, Vec(inputDim, 0));
        Ur = Mat(hiddenDim, Vec(hiddenDim, 0));
        Uz = Mat(hiddenDim, Vec(hiddenDim, 0));
        Ug = Mat(hiddenDim, Vec(hiddenDim, 0));
        Br = Vec(hiddenDim, 0);
        Bz = Vec(hiddenDim, 0);
        Bg = Vec(hiddenDim, 0);
        W = Mat(outputDim, Vec(hiddenDim, 0));
        B = Vec(outputDim, 0);
    }

    void zero()
    {
        for (std::size_t i = 0; i < Wr.size(); i++) {
            for (std::size_t j = 0; j < Wr[0].size(); j++) {
                Wr[i][j] = 0; Wz[i][j] = 0; Wg[i][j] = 0;
            }
        }
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                Ur[i][j] = 0; Uz[i][j] = 0; Ug[i][j] = 0;
            }
            Br[i] = 0; Bz[i] = 0; Bg[i] = 0;
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
        for (std::size_t i = 0; i < Wr.size(); i++) {
            for (std::size_t j = 0; j < Wr[0].size(); j++) {
                Wr[i][j] = uniform(Rand::engine);
                Wz[i][j] = uniform(Rand::engine);
                Wg[i][j] = uniform(Rand::engine);
            }
        }
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                Ur[i][j] = uniform(Rand::engine);
                Uz[i][j] = uniform(Rand::engine);
                Ug[i][j] = uniform(Rand::engine);
            }
            Br[i] = uniform(Rand::engine);
            Bz[i] = uniform(Rand::engine);
            Bg[i] = uniform(Rand::engine);
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

class GRU : public GRUParam
{
public:
    class State
    {
    public:
        Vec r;
        Vec z;
        Vec g;
        Vec h;
        Vec y;
    public:
        State(){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            r(Vec(hiddenDim, 0)), z(Vec(hiddenDim, 0)),
            g(Vec(hiddenDim, 0)), h(Vec(hiddenDim, 0)), y(Vec(outputDim, 0)){}
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
    struct Gain
    {
        double r;
        double z;
    };

protected:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    Gain g;
    Vec h;
    /* optimize */
    GRUParam d;
    GRUParam v;
    GRUParam s;
    double alpha_t;
    double beta_t;
    /* state */
    std::vector<State> states;
public:
    GRU(){}
    GRU(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    void clear();
    State feedForward(const Vec &x, const Vec &_h);
    void forward(const std::vector<Vec> &sequence);
    Vec forward(const Vec &x);
    void backward(const std::vector<Vec> &x, const std::vector<Vec> &E);
    void gradient(const std::vector<Vec> &x, const std::vector<Vec> &yt);
    void gradient(const std::vector<Vec> &x, const Vec &yt);
    void SGD(double learningRate = 0.001);
    void RMSProp(double learningRate = 0.001, double rho = 0.9);
    void Adam(double learningRate = 0.001, double alpha = 0.9, double beta = 0.99);
    static void test();
};

}
#endif // GRU_H
