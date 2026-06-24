#ifndef MAMBA_H
#define MAMBA_H
#include <memory>
#include <iostream>
#include "util.hpp"
#include "activate.h"
#include "optimize.h"
#include "loss.h"
#include "ilayer.h"

namespace RL {

/*
 * MambaLayer — Simplified Mamba (Selective State Space Model, S6)
 *
 * Core selective SSM with input-dependent parameters:
 *   B(x) = sigmoid(W_B · x + b_B)        — input-dependent input gate
 *   Δ(x) = softplus(W_Δ · x + b_Δ)       — input-dependent step size
 *   A: diagonal matrix (learned, clamped to (0, 1] for stability)
 *
 * Discretization (ZOH with diagonal A):
 *   Ā = exp(-Δ⊙(1 - A))                  — stable decay factor in (0, 1)
 *   B̄ = Δ ⊙ B(x)                         — simplified ZOH
 *
 * State update (per-element, diagonal A):
 *   h_i(t) = Ā_i · h_i(t-1) + B̄_i · x_i
 *
 * Output projection:
 *   y = tanh(C · h + b)
 *
 * The layer supports both inputDim → hiddenDim internal projection
 * (via weight matrix W_in) and direct use when inputDim == hiddenDim.
 *
 * Training: BPTT via the same cache+backward pattern as SSM/LSTM.
 *
 * Type registration: LAYER_MAMBA
 */
class MambaLayer : public iLayer
{
public:
    /* Per-timestep state for BPTT */
    class State
    {
    public:
        Tensor h;       // hidden state (hiddenDim × 1)
        Tensor y;       // output       (outputDim × 1)
        Tensor B;       // input gate   (hiddenDim × 1)  [cached for BPTT]
        Tensor delta;   // step size    (hiddenDim × 1)  [cached for BPTT]
        Tensor A_bar;   // decay factor (hiddenDim × 1)  [cached for BPTT]
        Tensor B_bar;   // effective B  (hiddenDim × 1)  [cached for BPTT]
    public:
        State() {}
        State(const State &r)
            : h(r.h), y(r.y), B(r.B), delta(r.delta),
              A_bar(r.A_bar), B_bar(r.B_bar) {}
        explicit State(std::size_t hiddenDim, std::size_t outputDim)
            : h(Tensor(hiddenDim, 1)), y(Tensor(outputDim, 1)),
              B(Tensor(hiddenDim, 1)), delta(Tensor(hiddenDim, 1)),
              A_bar(Tensor(hiddenDim, 1)), B_bar(Tensor(hiddenDim, 1)) {}

        void zero()
        {
            for (std::size_t k = 0; k < h.size(); k++) {
                h[k] = 0; B[k] = 0; delta[k] = 0;
                A_bar[k] = 0; B_bar[k] = 0;
            }
            for (std::size_t k = 0; k < y.size(); k++) y[k] = 0;
        }
    };

public:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;

    /* ——— Parameters ——— */

    /* Input projection (if inputDim != hiddenDim) */
    Tensor W_in;    // (hiddenDim × inputDim) or empty
    Tensor b_in;    // (hiddenDim × 1) or empty

    /* Diagonal A matrix (hiddenDim × 1) — stored as values in (0, 1] */
    Tensor A_diag;  // (hiddenDim × 1)

    /* Selective B: B(x) = sigmoid(W_B · x + b_B) */
    Tensor W_B;     // (hiddenDim × inputDim)
    Tensor b_B;     // (hiddenDim × 1)

    /* Selective Δ: Δ(x) = softplus(W_Δ · x + b_Δ) */
    Tensor W_delta; // (hiddenDim × inputDim)
    Tensor b_delta; // (hiddenDim × 1)

    /* Output projection: y = tanh(C · h + b) */
    Tensor C;       // (outputDim × hiddenDim)
    Tensor b;       // (outputDim × 1)

    /* Persistent hidden state */
    Tensor h;       // (hiddenDim × 1)

    /* Cached sequence states for BPTT */
    std::vector<State> states;
    std::vector<Tensor> cacheX;
    std::vector<Tensor> cacheE;

    /* ——— Gradients ——— */
    Tensor g_W_in, g_b_in;
    Tensor g_A_diag;
    Tensor g_W_B, g_b_B;
    Tensor g_W_delta, g_b_delta;
    Tensor g_C, g_b;

    /* ——— Optimizer state ——— */
    Tensor v_W_in, s_W_in;  // for Adam/RMSProp
    Tensor v_b_in, s_b_in;
    Tensor v_A_diag, s_A_diag;
    Tensor v_W_B, s_W_B;
    Tensor v_b_B, s_b_B;
    Tensor v_W_delta, s_W_delta;
    Tensor v_b_delta, s_b_delta;
    Tensor v_C, s_C;
    Tensor v_b, s_b;

public:
    MambaLayer() {}
    MambaLayer(const MambaLayer &r)
        : inputDim(r.inputDim), hiddenDim(r.hiddenDim), outputDim(r.outputDim),
          W_in(r.W_in), b_in(r.b_in), A_diag(r.A_diag),
          W_B(r.W_B), b_B(r.b_B),
          W_delta(r.W_delta), b_delta(r.b_delta),
          C(r.C), b(r.b), h(r.h), states(r.states) {}

    explicit MambaLayer(std::size_t inputDim_, std::size_t hiddenDim_,
                        std::size_t outputDim_, bool trainFlag);

    static std::shared_ptr<MambaLayer> _(std::size_t inputDim_, std::size_t hiddenDim_,
                                          std::size_t outputDim_, bool trainFlag)
    {
        return std::make_shared<MambaLayer>(inputDim_, hiddenDim_, outputDim_, trainFlag);
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
#endif // MAMBA_H
