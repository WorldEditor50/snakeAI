#ifndef MPG_H
#define MPG_H
#include "net.hpp"
#include "mamba.h"
#include "rl_basic.h"
#include "parameter.hpp"

namespace RL {

/*
 * MPG — Mamba Policy Gradient
 *
 * Policy gradient using MambaLayer (Selective State Space Model) as the
 * recurrent backbone.  Similar to DRPG (LSTM-based policy gradient) but
 * replaces LSTM with the Mamba S6 block.
 *
 * The MambaLayer provides input-dependent selective gating (B, Δ) and
 * a diagonal state-space dynamics that can capture temporal dependencies
 * more efficiently than LSTM.
 *
 * Architecture:
 *   MambaLayer(stateDim → hiddenDim → hiddenDim, tanh output)
 *   LayerNorm<Sigmoid, LN::Post>(hiddenDim → hiddenDim)
 *   Layer<Softmax>(hiddenDim → actionDim)
 */
class MPG
{
public:
    MPG(){}
    explicit MPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim);
    Tensor &eGreedyAction(const Tensor &state);
    RL::Tensor &noiseAction(const RL::Tensor &state);
    RL::Tensor &gumbelMax(const RL::Tensor &state);
    Tensor &action(const Tensor &state);
    void reinforce(std::vector<Step>& x, float learningRate);
    void reinforce1(std::vector<Step>& x, float learningRate);
    /* Reset persistent Mamba state to zero (call between independent episodes) */
    void resetState() { mamba_h.zero(); mamba->h.zero(); }
protected:
    std::size_t stateDim;
    std::size_t actionDim;
    float gamma;
    float exploringRate;
    float learningRate;
    float H0;
    GradValue alpha;
    std::shared_ptr<MambaLayer> mamba;
    Tensor mamba_h;     /* Mamba state after the last trajectory: saved before a
                           training replay and restored when the replay ends, so
                           inference keeps its temporal context. It is NOT read
                           by the action selectors — see the note in mpg.cpp. */
    Net policyNet;
};
}
#endif // MPG_H
