#include "gru.h"

RL::GRU::GRU(std::size_t inputDim_,
             std::size_t hiddenDim_,
             std::size_t outputDim_,
             bool trainFlag): GRUParam(inputDim_, hiddenDim_, outputDim_),
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
            h(t-1) -->----------------------------------
                        |                              |
                        |                              |
                        |                   ----(1-)---x
                        |                   |          |
                        |                   |          |
                  ------|-------------------x----------+
                  |     |                   |          |
                  |     x----------         |          h(t)
                  |__ g |        r |        | z
                        |          |        |
                        |       sigmoid  sigmoid
                        |          |        |
                        ---------------------
                        |
                        x(t)

        rt = sigmoid(Wr*xt + Ur*ht-1 + Br)
        zt = sigmoid(Wr*xt + Uz*ht-1 + Bz)
        gt = tanh(Wg*xt + Ug*(rt ⊙ ht-1) + Bg)
        ht = (1 - zt) ⊙ ht-1 + zt ⊙ gt
        yt = linear(W*ht + B)
    */
    State state(hiddenDim, outputDim);
    for (std::size_t i = 0; i < Wr.size(); i++) {
        for (std::size_t j = 0; j < Wr[0].size(); j++) {
            state.r[i] += Wr[i][j] * x[j];
            state.z[i] += Wz[i][j] * x[j];
            state.g[i] += Wg[i][j] * x[j];
        }
        for (std::size_t j = 0; j < Ur[0].size(); j++) {
            state.r[i] += Ur[i][j] * _h[j];
            state.z[i] += Uz[i][j] * _h[j];
        }
        state.r[i] = Sigmoid::_(state.r[i] + Br[i]);
        state.z[i] = Sigmoid::_(state.z[i] + Bz[i]);
        for (std::size_t j = 0; j < Ur[0].size(); j++) {
            state.g[i] += Ug[i][j] * _h[j] * state.r[j];
        }
        state.g[i] = Tanh::_(state.g[i] + Bg[i]);
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

void RL::GRU::backward(const std::vector<RL::Vec> &x, const std::vector<RL::Vec> &E)
{
    State delta(hiddenDim, outputDim);
    State delta_(hiddenDim, outputDim);
    for (int t = states.size() - 1; t >= 0; t--) {
        /* backward */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                delta.h[j] += W[i][j] * E[t][i];
            }
        }
#if 0
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                delta.h[j] += Ur[i][j] * delta_.r[i];
                delta.h[j] += Uz[i][j] * delta_.z[i];
                delta.h[j] += Ug[i][j] * delta_.g[i];
            }
        }
#else
        /* BPTT with EMA */
        double gamma = 0.99;
        for (std::size_t i = 0; i < Ur.size(); i++) {
            for (std::size_t j = 0; j < Ur[0].size(); j++) {
                delta.h[j] = delta.h[j]*gamma + Ur[i][j] * delta_.r[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Uz[i][j] * delta_.z[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Ug[i][j] * delta_.g[i]*(1 - gamma);
            }
        }
#endif
        /*
            dht/dzt = -ht-1 + gt
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
            δh = E + δht+1
            δg = δh ⊙ zt ⊙ dtanh(gt)
            δr = δh ⊙ (UTg * δg) ⊙ ht-1 ⊙ dsigmoid(rt)
            δz = δh ⊙ (gt - ht-1) ⊙ dsigmoid(zt)

        */
        Vec _h = t > 0 ? states[t - 1].h : Vec(hiddenDim, 0);
        for (std::size_t i = 0; i < Ug.size(); i++) {
            delta.g[i] = delta.h[i] * states[t].z[i] * Tanh::d(states[t].g[i]);
            delta.z[i] = delta.h[i] * (states[t].g[i] - _h[i]) * Sigmoid::d(states[t].z[i]);
        }
        Vec dhr(hiddenDim, 0);
        for (std::size_t i = 0; i < Ug.size(); i++) {
            for (std::size_t j = 0; j < Ug[0].size(); j++) {
                dhr[j] += Ug[i][j] * states[t].g[i];
            }
        }
        for (std::size_t i = 0; i < Ug.size(); i++) {
            delta.r[i] = delta.h[i] * dhr[i] * _h[i] *Sigmoid::d(states[t].r[i]);
        }
        /* gradient */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                d.W[i][j] += E[t][i] * Linear::d(states[t].y[i]) * states[t].h[j];
            }
            d.B[i] += E[t][i] * Linear::d(states[t].y[i]);
        }
        for (std::size_t i = 0; i < Wr.size(); i++) {
            for (std::size_t j = 0; j < Wr[0].size(); j++) {
                d.Wr[i][j] += delta.r[i] * x[t][j];
                d.Wz[i][j] += delta.z[i] * x[t][j];
                d.Wg[i][j] += delta.g[i] * x[t][j];
            }
        }
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

void RL::GRU::gradient(const std::vector<RL::Vec> &x, const std::vector<RL::Vec> &yt)
{
    /* loss */
    std::vector<RL::Vec> E(yt.size(), Vec(outputDim, 0));
    for (int t = states.size() - 1; t >= 0; t--) {
        for (std::size_t i = 0; i < outputDim; i++) {
            E[t][i] = 2 * (states[t].y[i] - yt[t][i]);
        }
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::GRU::gradient(const std::vector<RL::Vec> &x, const RL::Vec &yt)
{
    /* loss */
    std::vector<RL::Vec> E(states.size(), Vec(outputDim, 0));
    int t = states.size() - 1;
    for (std::size_t i = 0; i < outputDim; i++) {
        E[t][i] = 2 * (states[t].y[i] - yt[i]);
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::GRU::SGD(double learningRate)
{
    Optimizer::SGD(d.W, W, learningRate);
    Optimizer::SGD(d.B, B, learningRate);

    Optimizer::SGD(d.Wr, Wr, learningRate);
    Optimizer::SGD(d.Wz, Wz, learningRate);
    Optimizer::SGD(d.Wg, Wg, learningRate);

    Optimizer::SGD(d.Ur, Ur, learningRate);
    Optimizer::SGD(d.Uz, Uz, learningRate);
    Optimizer::SGD(d.Ug, Ug, learningRate);

    Optimizer::SGD(d.Br, Br, learningRate);
    Optimizer::SGD(d.Bz, Bz, learningRate);
    Optimizer::SGD(d.Bg, Bg, learningRate);
    d.zero();
    return;
}

void RL::GRU::RMSProp(double learningRate, double rho)
{
    Optimizer::RMSProp(d.W, s.W, W, learningRate, rho);
    Optimizer::RMSProp(d.B, s.B, B, learningRate, rho);

    Optimizer::RMSProp(d.Wr, s.Wr, Wr, learningRate, rho);
    Optimizer::RMSProp(d.Wz, s.Wz, Wz, learningRate, rho);
    Optimizer::RMSProp(d.Wg, s.Wg, Wg, learningRate, rho);

    Optimizer::RMSProp(d.Ur, s.Ur, Ur, learningRate, rho);
    Optimizer::RMSProp(d.Uz, s.Uz, Uz, learningRate, rho);
    Optimizer::RMSProp(d.Ug, s.Ug, Ug, learningRate, rho);

    Optimizer::RMSProp(d.Br, s.Br, Br, learningRate, rho);
    Optimizer::RMSProp(d.Bz, s.Bz, Bz, learningRate, rho);
    Optimizer::RMSProp(d.Bg, s.Bg, Bg, learningRate, rho);

    d.zero();
    return;
}

void RL::GRU::Adam(double learningRate,  double alpha, double beta)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimizer::Adam(d.W, s.W, v.W, W, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.B, s.B, v.B, B, alpha_t, beta_t, learningRate, alpha, beta);

    Optimizer::Adam(d.Wr, s.Wr, v.Wr, Wr, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Wz, s.Wz, v.Wz, Wz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Wg, s.Wg, v.Wg, Wg, alpha_t, beta_t, learningRate, alpha, beta);

    Optimizer::Adam(d.Ur, s.Ur, v.Ur, Ur, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Uz, s.Uz, v.Uz, Uz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Ug, s.Ug, v.Ug, Ug, alpha_t, beta_t, learningRate, alpha, beta);

    Optimizer::Adam(d.Br, s.Br, v.Br, Br, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Bz, s.Bz, v.Bz, Bz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimizer::Adam(d.Bg, s.Bg, v.Bg, Bg, alpha_t, beta_t, learningRate, alpha, beta);
    d.zero();
    return;
}

void RL::GRU::test()
{
    GRU gru(2, 8, 1, true);
    auto zeta = [](double x, double y) -> double {
        return x*x + y*y + x*y;
    };
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<Vec> data;
    std::vector<Vec> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Vec p(2);
            double x = uniform(Rand::engine);
            double y = uniform(Rand::engine);
            double z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Vec q(1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Vec> &batchData,
            std::vector<Vec> &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = selectIndex(Rand::engine);
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Vec> batchData;
        std::vector<Vec> batchTarget;
        sample(batchData, batchTarget, 32);
        gru.forward(batchData);
        gru.gradient(batchData, batchTarget);
        gru.Adam(0.001);
    }

    gru.clear();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Vec p(2);
            double x = uniform(Rand::engine);
            double y = uniform(Rand::engine);
            double z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            auto s = gru.forward(p);
            std::cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
