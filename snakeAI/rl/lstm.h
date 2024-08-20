#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <memory>
#include "util.hpp"
#include "activate.h"
#include "optimize.h"
#include "loss.h"
#include "ilayer.h"

namespace RL {

class LSTMParam
{
public:
    /* input gate */
    Tensor wi;
    Tensor ui;
    Tensor bi;
    /* generate */
    Tensor wg;
    Tensor ug;
    Tensor bg;
    /* forget gate */
    Tensor wf;
    Tensor uf;
    Tensor bf;
    /* output gate */
    Tensor wo;
    Tensor uo;
    Tensor bo;
    /* predict */
    Tensor w;
    Tensor b;
public:
    LSTMParam(){}
    LSTMParam(const LSTMParam &r)
        :wi(r.wi), ui(r.ui), bi(r.bi),
         wg(r.wg), ug(r.ug), bg(r.bg),
         wf(r.wf), uf(r.uf), bf(r.bf),
         wo(r.wo), uo(r.uo), bo(r.bo),
         w(r.w), b(r.b){}
    explicit LSTMParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        wi = Tensor(hiddenDim, inputDim);
        wg = Tensor(hiddenDim, inputDim);
        wf = Tensor(hiddenDim, inputDim);
        wo = Tensor(hiddenDim, inputDim);

        ui = Tensor(hiddenDim, hiddenDim);
        ug = Tensor(hiddenDim, hiddenDim);
        uf = Tensor(hiddenDim, hiddenDim);
        uo = Tensor(hiddenDim, hiddenDim);

        bi = Tensor(hiddenDim, 1);
        bg = Tensor(hiddenDim, 1);
        bf = Tensor(hiddenDim, 1);
        bo = Tensor(hiddenDim, 1);

        w = Tensor(outputDim, hiddenDim);
        b = Tensor(outputDim, 1);
    }
    void zero()
    {
        std::vector<Tensor*> weights = {&wi, &wg, &wf, &wo,
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
        std::vector<Tensor*> weights = {&wi, &wg, &wf, &wo,
                                     &ui, &ug, &uf, &uo,
                                     &bi, &bg, &bf, &bo,
                                     &w, &b};
        for (std::size_t i = 0; i < weights.size(); i++) {
            RL::uniformRand(*weights[i], -1, 1);
        }
        return;
    }
};

class LSTM : public iLayer
{
public:
    class State
    {
    public:
        Tensor i;
        Tensor f;
        Tensor g;
        Tensor o;
        Tensor c;
        Tensor h;
        Tensor y;
    public:
        State(){}
        State(const State &r)
            :i(r.i),f(r.f), g(r.g),
             o(r.o),c(r.c),h(r.h), y(r.y){}
        explicit State(std::size_t hiddenDim, std::size_t outputDim):
            i(Tensor(hiddenDim, 1)),f(Tensor(hiddenDim, 1)),g(Tensor(hiddenDim, 1)),
            o(Tensor(hiddenDim, 1)),c(Tensor(hiddenDim, 1)),h(Tensor(hiddenDim, 1)),
            y(Tensor(outputDim, 1)){}

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

    /* input gate */
    Tensor wi;
    Tensor ui;
    Tensor bi;
    /* generate */
    Tensor wg;
    Tensor ug;
    Tensor bg;
    /* forget gate */
    Tensor wf;
    Tensor uf;
    Tensor bf;
    /* output gate */
    Tensor wo;
    Tensor uo;
    Tensor bo;
    /* predict */
    Tensor w;
    Tensor b;

    Tensor h;
    Tensor c;
    /* state */
    std::vector<State> states;

    std::vector<Tensor> cacheX;
    std::vector<Tensor> cacheE;
protected:
    LSTMParam g;
    LSTMParam v;
    LSTMParam s;
public:
    LSTM(){}
    LSTM(const LSTM &r)
        :inputDim(r.inputDim),hiddenDim(r.hiddenDim),outputDim(r.outputDim),
          wi(r.wi), ui(r.ui), bi(r.bi),
          wg(r.wg), ug(r.ug), bg(r.bg),
          wf(r.wf), uf(r.uf), bf(r.bf),
          wo(r.wo), uo(r.uo), bo(r.bo),
          w(r.w), b(r.b),
          h(r.h),c(r.c),states(r.states),
          g(r.g),v(r.v),s(r.s){}
    explicit LSTM(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    static std::shared_ptr<LSTM> _(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag)
    {
        return std::make_shared<LSTM>(inputDim_, hiddenDim_, outputDim_, trainFlag);
    }
    void reset();
    /* forward */
    State feedForward(const Tensor &x, const Tensor &_h, const Tensor &_c);
    Tensor &forward(const Tensor &x, bool inference=false) override;
    /* backward */
    void backwardAtTime(int t, const Tensor &x, const Tensor &E, State &delta_);
    void backward(const std::vector<Tensor> &x, const std::vector<Tensor> &E);
    void cacheError(const Tensor &e) override;
    /* optimize */
    void SGD(float lr) override;
    virtual void RMSProp(float lr, float rho, float decay, bool clipGrad) override;
    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override;
    void clamp(float c0, float cn);
    /* parameter */
    virtual void copyTo(iLayer* layer);
    virtual void softUpdateTo(iLayer* layer, float rho);
    virtual void write(std::ofstream &file) override;
    virtual void read(std::ifstream &file) override;
    static void test();
};

}
#endif // LSTM_H
