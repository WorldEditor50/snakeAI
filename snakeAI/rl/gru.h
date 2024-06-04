#ifndef GRU_H
#define GRU_H
#include <iostream>
#include "util.hpp"
#include "activate.h"
#include "optimize.h"
#include "loss.h"

namespace RL {

class GRUParam
{
public:
    /* reset gate */
    Tensor Wr;
    Tensor Ur;
    Tensor Br;
    /* update gate */
    Tensor Wz;
    Tensor Uz;
    Tensor Bz;
    /* h */
    Tensor Wg;
    Tensor Ug;
    Tensor Bg;
    /* predict */
    Tensor W;
    Tensor B;
public:
    GRUParam(){}
    GRUParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wr = Tensor(hiddenDim, inputDim);
        Wz = Tensor(hiddenDim, inputDim);
        Wg = Tensor(hiddenDim, inputDim);
        Ur = Tensor(hiddenDim, hiddenDim);
        Uz = Tensor(hiddenDim, hiddenDim);
        Ug = Tensor(hiddenDim, hiddenDim);
        Br = Tensor(hiddenDim, 1);
        Bz = Tensor(hiddenDim, 1);
        Bg = Tensor(hiddenDim, 1);
        W = Tensor(outputDim, hiddenDim);
        B = Tensor(outputDim, 1);
    }

    void zero()
    {
        std::vector<Tensor*> weights = {&Wr, &Wz, &Wg,
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
        std::vector<Tensor*> weights = {&Wr, &Wz, &Wg,
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
        Tensor r;
        Tensor z;
        Tensor g;
        Tensor h;
        Tensor y;
    public:
        State(){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            r(Tensor(hiddenDim, 1)), z(Tensor(hiddenDim, 1)),
            g(Tensor(hiddenDim, 1)), h(Tensor(hiddenDim, 1)), y(Tensor(outputDim, 1)){}
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
    Tensor h;
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
    State feedForward(const Tensor &x, const Tensor &_h);
    void forward(const std::vector<Tensor> &sequence);
    Tensor forward(const Tensor &x);
    void backward(const std::vector<Tensor> &x, const std::vector<Tensor> &E);
    void gradient(const std::vector<Tensor> &x, const std::vector<Tensor> &yt);
    void gradient(const std::vector<Tensor> &x, const Tensor &yt);
    void SGD(float learningRate = 0.001);
    void RMSProp(float learningRate = 0.001, float rho = 0.9);
    void Adam(float learningRate = 0.001, float alpha = 0.9, float beta = 0.99);
    static void test();
};

}
#endif // GRU_H
