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
        auto uniform = []()->double{
            int r1 = rand()%10;
            int r2 = rand()%10;
            double s = r1 > r2 ? 1 : -1;
            return s * double(rand()%10000) / 10000;
        };
        for (std::size_t i = 0; i < Wr.size(); i++) {
            for (std::size_t j = 0; j < Wr[0].size(); j++) {
                Wr[i][j] = uniform(); Wz[i][j] = uniform(); Wg[i][j] = uniform();
            }
        }
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                Ur[i][j] = uniform(); Uz[i][j] = uniform(); Ug[i][j] = uniform();
            }
            Br[i] = uniform(); Bz[i] = uniform(); Bg[i] = uniform();
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = uniform();
            }
            B[i] = uniform();
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
            r(Vec(hiddenDim, 0)),z(Vec(hiddenDim, 0)),
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
protected:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
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
    void gradient(const std::vector<Vec> &x, const std::vector<Vec> &yt);
    void SGD(double learningRate);
    void RMSProp(double learningRate, double rho);
    static void test();
};

}
#endif // GRU_H
