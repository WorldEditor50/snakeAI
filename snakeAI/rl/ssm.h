#ifndef SSM_H
#define SSM_H
#include <memory>
#include <iostream>
#include "util.hpp"
#include "activate.h"
#include "optimize.h"
#include "loss.h"
#include "ilayer.h"

namespace RL {

/*
 * SSMParam — Parameter bundle for SSM (State Space Model)
 *
 * A simple discrete-time linear state space model:
 *   h(t) = A·h(t-1) + B·x(t)
 *   y(t) = tanh(C·h(t) + b)
 */
class SSMParam
{
public:
    Tensor A;   // state transition (hiddenDim × hiddenDim)
    Tensor B;   // input projection (hiddenDim × inputDim)
    Tensor C;   // output projection (outputDim × hiddenDim)
    Tensor b;   // output bias      (outputDim × 1)

public:
    SSMParam() {}
    SSMParam(const SSMParam &r)
        : A(r.A), B(r.B), C(r.C), b(r.b) {}
    explicit SSMParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        A = Tensor(hiddenDim, hiddenDim);
        B = Tensor(hiddenDim, inputDim);
        C = Tensor(outputDim, hiddenDim);
        b = Tensor(outputDim, 1);
    }

    void zero()
    {
        A.zero(); B.zero(); C.zero(); b.zero();
        return;
    }

    void random()
    {
        Random::uniform(A, -1, 1);
        Random::uniform(B, -1, 1);
        Random::uniform(C, -1, 1);
        Random::uniform(b, -1, 1);
        return;
    }
};

/*
 * SSM — Simple State Space Model (iLayer)
 *
 * A discretised linear SSM that processes sequential data:
 *
 *   h(t) = A·h(t-1) + B·x(t)
 *   y(t) = tanh(C·h(t) + b)
 *
 * Training uses BPTT (Backpropagation Through Time) following the
 * same cache+backward pattern as LSTM:
 *   1. forward(x): caches (x, state) pairs
 *   2. cacheError(e): collects loss gradients at each step
 *   3. optimizer (e.g. RMSProp): calls backward() then updates weights
 *
 * Type registration: LAYER_SSM
 */
class SSM : public iLayer
{
public:
    /* Per-timestep state for BPTT */
    class State
    {
    public:
        Tensor h;   // hidden state (hiddenDim × 1)
        Tensor y;   // output       (outputDim × 1)
    public:
        State() {}
        State(const State &r)
            : h(r.h), y(r.y) {}
        explicit State(std::size_t hiddenDim, std::size_t outputDim)
            : h(Tensor(hiddenDim, 1)), y(Tensor(outputDim, 1)) {}

        void zero()
        {
            for (std::size_t k = 0; k < h.size(); k++) h[k] = 0;
            for (std::size_t k = 0; k < y.size(); k++) y[k] = 0;
            return;
        }
    };

public:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;

    /* Parameters */
    Tensor A;   // state transition (hiddenDim × hiddenDim)
    Tensor B;   // input projection (hiddenDim × inputDim)
    Tensor C;   // output projection (outputDim × hiddenDim)
    Tensor b;   // output bias      (outputDim × 1)

    /* Persistent hidden state */
    Tensor h;

    /* Cached sequence states for BPTT */
    std::vector<State> states;
    std::vector<Tensor> cacheX;
    std::vector<Tensor> cacheE;

    /* Gradients (public for testing) */
    SSMParam g;
    SSMParam v;
    SSMParam s;

public:
    SSM() {}
    SSM(const SSM &r)
        : inputDim(r.inputDim), hiddenDim(r.hiddenDim), outputDim(r.outputDim),
          A(r.A), B(r.B), C(r.C), b(r.b), h(r.h), states(r.states),
          g(r.g), v(r.v), s(r.s) {}

    explicit SSM(std::size_t inputDim_, std::size_t hiddenDim_,
                 std::size_t outputDim_, bool trainFlag);

    static std::shared_ptr<SSM> _(std::size_t inputDim_, std::size_t hiddenDim_,
                                  std::size_t outputDim_, bool trainFlag)
    {
        return std::make_shared<SSM>(inputDim_, hiddenDim_, outputDim_, trainFlag);
    }

    /* Lifecycle */
    void reset();
    void initParams() override;

    /* Forward — single timestep */
    State feedForward(const Tensor &x, const Tensor &_h);
    Tensor &forward(const Tensor &x, bool inference = false) override;

    /* Backward via BPTT */
    void backwardAtTime(int t, const Tensor &x, const Tensor &E, State &delta_);
    void backward(const std::vector<Tensor> &x, const std::vector<Tensor> &E);
    void cacheError(const Tensor &e) override;

    /* Optimizers */
    void SGD(float lr) override;
    void RMSProp(float lr, float rho, float decay, bool clipGrad) override;
    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override;
    void clamp(float c0, float cn) override;

    /* Parameter ops */
    void copyTo(iLayer *layer) override;
    void softUpdateTo(iLayer *layer, float alpha) override;
    void write(std::ofstream &file) override;
    void read(std::ifstream &file) override;

    /* Test */
    static void test();
};

} // namespace RL
#endif // SSM_H
