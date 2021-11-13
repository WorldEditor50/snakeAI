#include "gru.h"

RL::GRU::GRU(std::size_t inputDim_,
             std::size_t hiddenDim_,
             std::size_t outputDim_,
             bool trainFlag):
  GRUParam(inputDim_, hiddenDim_, outputDim_),
  inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    if (trainFlag == true) {
      d = GRUParam(inputDim_, hiddenDim_, outputDim_);
      v = GRUParam(inputDim_, hiddenDim_, outputDim_);
      s = GRUParam(inputDim_, hiddenDim_, outputDim_);
    }
    h = Vec(hiddenDim, 0);
    alpha_t = 1;
    beta_t = 1;
    GRUParam::random();
}

void RL::GRU::clear()
{
    h.assign(hiddenDim, 0);
    return;
}

RL::GRU::State RL::GRU::feedForward(const RL::Vec &x, const RL::Vec &_h)
{
    /*
        rt = sigmoid(Wr*xt + Ur*ht-1 + Br)
        zt = sigmoid(Wr*xt + Uz*ht-1 + Bz)
        gt = tanh(Wg*xt + Ug*(rt ⊙ ht-1) + Bg)
        ht = (1 - zt) ⊙ ht-1 + zt ⊙ gt
        yt = sigmoid(W*ht + B)
    */
    State state(hiddenDim, outputDim);
    for (std::size_t i = 0; i < Wr.size(); i++) {
        double sr = 0;
        double sz = 0;
        double sg = 0;
        for (std::size_t j = 0; j < Wr[0].size(); j++) {
            sr += Wr[i][j] * x[j];
            sz += Wz[i][j] * x[j];
            sg += Wg[i][j] * x[j];
        }
        for (std::size_t j = 0; j < Ur[0].size(); j++) {
            sr += Ur[i][j] * _h[j];
            sz += Uz[i][j] * _h[j];
        }
        state.r[i] = Sigmoid::_(sr + Br[i]);
        state.z[i] = Sigmoid::_(sz + Bz[i]);
        for (std::size_t j = 0; j < Ur[0].size(); j++) {
            sg += Ug[i][j] * _h[j] * state.r[j];
        }
        state.g[i] = Tanh::_(sg + Bg[i]);
        state.h[i] = (1 - state.z[i]) * _h[i] + state.z[i] * state.g[i];
    }

    for (std::size_t i = 0; i < W.size(); i++) {
        double sy = 0;
        for (std::size_t j = 0; j < W[0].size(); j++) {
            sy += W[i][j] * state.h[j];
        }
        state.y[i] = Linear::_(sy + B[i]);
    }
    return state;
}

void RL::GRU::forward(const std::vector<RL::Vec> &sequence)
{
    h.assign(hiddenDim, 0);
    for (auto &x : sequence) {
        State state = feedForward(x, h);
        h = state.h;
        states.push_back(state);
    }
    return;
}

RL::Vec RL::GRU::forward(const RL::Vec &x)
{
    State state = feedForward(x, h);
    h = state.h;
    return state.y;
}

void RL::GRU::gradient(const std::vector<RL::Vec> &x, const std::vector<RL::Vec> &yt)
{
    State delta(hiddenDim, outputDim);
    State delta_(hiddenDim, outputDim);
    for (int t = states.size() - 1; t >= 0; t--) {
        /* loss */
        for (std::size_t i = 0; i < yt.size(); i++) {
            delta.y[i] = 2 * (states[t].y[i] - yt[t][i]);
        }
        /* backward */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                delta.h[j] += W[i][j] * delta.y[i];
            }
        }

        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                delta.h[j] += Ur[i][j] * delta_.r[i];
                delta.h[j] += Uz[i][j] * delta_.z[i];
                delta.h[j] += Ug[i][j] * delta_.g[i];
            }
        }
        /*
            dht/dzt = ht-1 + gt
            dht/dWz = (dht/dzt ⊙ zt ⊙ (1 - zt))*xTt
            dht/dUz = (dht/dzt ⊙ zt ⊙ (1 - zt))*hTt-1
            dht/dWg = (zt ⊙ (1 - gt^2) * xTt
            dht/dUg = (zt ⊙ (1 - gt^2) * (rt ⊙ ht-1)T
            dht/drt = (UTg*(zt ⊙ (1 - gt^2))) ⊙ ht-1
            dht/dWr = (dht/drt ⊙ rt ⊙ (1 - rt))*xTt
            dht/dUr = (dht/drt ⊙ rt ⊙ (1 - rt))*hTt-1

            dht/dxt = WTz * ((gt - ht-1) ⊙ zt ⊙ (1 - zt)) +
                      zt ⊙ (WTg*(1 - gt^2) + WTr*((UTg*(1 - gt^2)) ⊙ ht-1 ⊙ rt ⊙ (1 - rt)))

            dht/dht-1 = -UTz * (ht-1 ⊙ zt ⊙ (1 - zt)) + (1 - zt) + UTz * (gt ⊙ zt ⊙ (1 - zt)) +
                        UTr * ((UTg*(zt ⊙ (1 - gt^2))) ⊙ ht-1 ⊙ rt ⊙ (1 -rt) +
                        (UTg*(zt ⊙ (1 - gt^2))) ⊙ rt
            --------------------------------------------------
            δg = zt ⊙ dtanh(gt)
            δr = (UTg * δg) ⊙ ht-1 ⊙ dsigmoid(rt)
            δz = (ht-1 + gt) ⊙ dsigmoid(zt)

        */
        for (std::size_t i = 0; i < Ug.size(); i++) {
            delta.g[i] = states[t].z[i] * Tanh::d(states[t].g[i]);
            delta.z[i] = (states[t - 1].h[i] + states[t].g[i]) * Sigmoid::d(states[t].z[i]);
        }
        Vec dhr(hiddenDim, 0);
        for (std::size_t i = 0; i < Ug.size(); i++) {
            for (std::size_t j = 0; j < Ug[0].size(); j++) {
                dhr[j] += Ug[i][j] * states[t].g[i];
            }
        }
        for (std::size_t i = 0; i < Ug.size(); i++) {
            delta.r[i] = dhr[i] * states[t - 1].h[i] *Sigmoid::d(states[t].r[i]);
        }

        /* gradient */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                d.W[i][j] += delta.y[i] * Linear::d(states[t].y[i]) * states[t].h[j];
            }
            d.B[i] += delta.y[i] * Linear::d(states[t].y[i]);
        }
        for (std::size_t i = 0; i < Wr.size(); i++) {
            for (std::size_t j = 0; j < Wr[0].size(); j++) {
                d.Wr[i][j] += delta.r[i] * x[t][j];
                d.Wz[i][j] += delta.z[i] * x[t][j];
                d.Wg[i][j] += delta.g[i] * x[t][j];
            }
        }
        Vec _h = t > 0 ? states[t - 1].h : Vec(hiddenDim, 0);
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                d.Ur[i][j] += delta.r[i] * _h[j];
                d.Uz[i][j] += delta.z[i] * _h[j];
                d.Ug[i][j] += delta.g[i] * states[t].r[j] * _h[j];
            }
        }
        for (std::size_t i = 0; i < Br.size(); i++) {
            d.Br[i] += delta.r[i];
            d.Bz[i] += delta.z[i];
            d.Bg[i] += delta.g[i];
        }
        /* next */
        delta_ = delta;
        delta.zero();
    }
    states.clear();
    return;
}

void RL::GRU::RMSProp(double learningRate, double rho)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            s.W[i][j] = rho * s.W[i][j] + (1 - rho) * d.W[i][j] * d.W[i][j];
            W[i][j] -= learningRate * d.W[i][j] / (sqrt(s.W[i][j]) + 1e-9);
        }
        s.B[i] = rho * s.B[i] + (1 - rho) * d.B[i] * d.B[i];
        B[i] -= learningRate * d.B[i] / (sqrt(s.B[i]) + 1e-9);
    }
    for (std::size_t i = 0; i < Wr.size(); i++) {
        for (std::size_t j = 0; j < Wr[0].size(); j++) {
            s.Wr[i][j] = rho * s.Wr[i][j] + (1 - rho) * d.Wr[i][j] * d.Wr[i][j];
            s.Wz[i][j] = rho * s.Wz[i][j] + (1 - rho) * d.Wz[i][j] * d.Wz[i][j];
            s.Wg[i][j] = rho * s.Wg[i][j] + (1 - rho) * d.Wg[i][j] * d.Wg[i][j];
            Wr[i][j] -= learningRate * d.Wr[i][j] / (sqrt(s.Wr[i][j]) + 1e-9);
            Wz[i][j] -= learningRate * d.Wz[i][j] / (sqrt(s.Wz[i][j]) + 1e-9);
            Wg[i][j] -= learningRate * d.Wg[i][j] / (sqrt(s.Wg[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Ur.size(); i++) {
        for (std::size_t j = 0; j < Ur[0].size(); j++) {
            s.Ur[i][j] = rho * s.Ur[i][j] + (1 - rho) * d.Ur[i][j] * d.Ur[i][j];
            s.Uz[i][j] = rho * s.Uz[i][j] + (1 - rho) * d.Uz[i][j] * d.Uz[i][j];
            s.Ug[i][j] = rho * s.Ug[i][j] + (1 - rho) * d.Ug[i][j] * d.Ug[i][j];
            Ur[i][j] -= learningRate * d.Ur[i][j] / (sqrt(s.Ur[i][j]) + 1e-9);
            Uz[i][j] -= learningRate * d.Uz[i][j] / (sqrt(s.Uz[i][j]) + 1e-9);
            Ug[i][j] -= learningRate * d.Ug[i][j] / (sqrt(s.Ug[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Br.size(); i++) {
        s.Br[i] = rho * s.Br[i] + (1 - rho) * d.Br[i] * d.Br[i];
        s.Bz[i] = rho * s.Bz[i] + (1 - rho) * d.Bz[i] * d.Bz[i];
        s.Bg[i] = rho * s.Bg[i] + (1 - rho) * d.Bg[i] * d.Bg[i];
        Br[i] -= learningRate * d.Br[i] / (sqrt(s.Br[i]) + 1e-9);
        Bz[i] -= learningRate * d.Bz[i] / (sqrt(s.Bz[i]) + 1e-9);
        Bg[i] -= learningRate * d.Bg[i] / (sqrt(s.Bg[i]) + 1e-9);
    }
    d.zero();
    return;
}
