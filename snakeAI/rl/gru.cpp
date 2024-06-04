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
    h = Tensor(hiddenDim, 1);
    alpha_t = 1;
    beta_t = 1;
    GRUParam::random();
}

void RL::GRU::clear()
{
    h.zero();
    return;
}

RL::GRU::State RL::GRU::feedForward(const RL::Tensor &x, const RL::Tensor &_h)
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
    for (std::size_t i = 0; i < Wr.shape[0]; i++) {
        for (std::size_t j = 0; j < Wr.shape[1]; j++) {
            state.r[i] += Wr(i, j) * x[j];
            state.z[i] += Wz(i, j) * x[j];
            state.g[i] += Wg(i, j) * x[j];
        }
        for (std::size_t j = 0; j < Ur.shape[1]; j++) {
            state.r[i] += Ur(i, j) * _h[j];
            state.z[i] += Uz(i, j) * _h[j];
        }
        state.r[i] = Sigmoid::f(state.r[i] + Br[i]);
        state.z[i] = Sigmoid::f(state.z[i] + Bz[i]);
        for (std::size_t j = 0; j < Ur.shape[1]; j++) {
            state.g[i] += Ug(i, j) * _h[j] * state.r[j];
        }
        state.g[i] = Tanh::f(state.g[i] + Bg[i]);
        state.h[i] = (1 - state.z[i]) * _h[i] + state.z[i] * state.g[i];
    }

    for (std::size_t i = 0; i < W.shape[0]; i++) {
        float sy = 0;
        for (std::size_t j = 0; j < W.shape[1]; j++) {
            sy += W(i, j) * state.h[j];
        }
        state.y[i] = Linear::f(sy + B[i]);
    }
    return state;
}

void RL::GRU::forward(const std::vector<RL::Tensor> &sequence)
{
    h.zero();
    for (auto &x : sequence) {
        State state = feedForward(x, h);
        h = state.h;
        states.push_back(state);
    }
    return;
}

RL::Tensor RL::GRU::forward(const RL::Tensor &x)
{
    State state = feedForward(x, h);
    h = state.h;
    return state.y;
}

void RL::GRU::backward(const std::vector<RL::Tensor> &x, const std::vector<RL::Tensor> &E)
{
    State delta(hiddenDim, outputDim);
    State delta_(hiddenDim, outputDim);
    for (int t = states.size() - 1; t >= 0; t--) {
        /* backward */
        for (std::size_t i = 0; i < W.shape[0]; i++) {
            for (std::size_t j = 0; j < W.shape[1]; j++) {
                delta.h[j] += W(i, j) * E[t][i];
            }
        }
#if 0
        for (std::size_t i = 0; i < Ur.shape[0]; i++) {
            for (std::size_t j = 0; j < Ur.shape[1]; j++) {
                delta.h[j] += Ur(i, j) * delta_.r[i];
                delta.h[j] += Uz(i, j) * delta_.z[i];
                delta.h[j] += Ug(i, j) * delta_.g[i];
            }
        }
#else
        /* BPTT with EMA */
        float gamma = 0.99;
        for (std::size_t i = 0; i < Ur.shape[0]; i++) {
            for (std::size_t j = 0; j < Ur.shape[1]; j++) {
                delta.h[j] = delta.h[j]*gamma + Ur(i, j) * delta_.r[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Uz(i, j) * delta_.z[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Ug(i, j) * delta_.g[i]*(1 - gamma);
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
        Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
        for (std::size_t i = 0; i < Ug.shape[0]; i++) {
            delta.g[i] = delta.h[i] * states[t].z[i] * Tanh::d(states[t].g[i]);
            delta.z[i] = delta.h[i] * (states[t].g[i] - _h[i]) * Sigmoid::d(states[t].z[i]);
        }
        Tensor dhr(hiddenDim, 1);
        for (std::size_t i = 0; i < Ug.shape[0]; i++) {
            for (std::size_t j = 0; j < Ug.shape[1]; j++) {
                dhr[j] += Ug(i, j) * states[t].g[i];
            }
        }
        for (std::size_t i = 0; i < Ug.shape[0]; i++) {
            delta.r[i] = delta.h[i] * dhr[i] * _h[i] *Sigmoid::d(states[t].r[i]);
        }
        /* gradient */
        for (std::size_t i = 0; i < W.shape[0]; i++) {
            for (std::size_t j = 0; j < W.shape[1]; j++) {
                d.W(i, j) += E[t][i] * Linear::d(states[t].y[i]) * states[t].h[j];
            }
            d.B[i] += E[t][i] * Linear::d(states[t].y[i]);
        }
        for (std::size_t i = 0; i < Wr.shape[0]; i++) {
            for (std::size_t j = 0; j < Wr.shape[1]; j++) {
                d.Wr(i, j) += delta.r[i] * x[t][j];
                d.Wz(i, j) += delta.z[i] * x[t][j];
                d.Wg(i, j) += delta.g[i] * x[t][j];
            }
        }
        for (std::size_t i = 0; i < Ur.shape[0]; i++) {
            for (std::size_t j = 0; j < Ur.shape[1]; j++) {
                d.Ur(i, j) += delta.r[i] * _h[j];
                d.Uz(i, j) += delta.z[i] * _h[j];
                d.Ug(i, j) += delta.g[i] * states[t].r[j] * _h[j];
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

void RL::GRU::gradient(const std::vector<RL::Tensor> &x, const std::vector<RL::Tensor> &yt)
{
    /* loss */
    std::vector<RL::Tensor> E(yt.size(), Tensor(outputDim, 1));
    for (int t = states.size() - 1; t >= 0; t--) {
        for (std::size_t i = 0; i < outputDim; i++) {
            E[t][i] = 2 * (states[t].y[i] - yt[t][i]);
        }
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::GRU::gradient(const std::vector<RL::Tensor> &x, const RL::Tensor &yt)
{
    /* loss */
    std::vector<RL::Tensor> E(states.size(), Tensor(outputDim, 1));
    int t = states.size() - 1;
    for (std::size_t i = 0; i < outputDim; i++) {
        E[t][i] = 2 * (states[t].y[i] - yt[i]);
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::GRU::SGD(float learningRate)
{
    Optimize::SGD(W, d.W, learningRate);
    Optimize::SGD(B, d.B, learningRate);

    Optimize::SGD(Wr, d.Wr, learningRate);
    Optimize::SGD(Wz, d.Wz, learningRate);
    Optimize::SGD(Wg, d.Wg, learningRate);

    Optimize::SGD(Ur, d.Ur, learningRate);
    Optimize::SGD(Uz, d.Uz, learningRate);
    Optimize::SGD(Ug, d.Ug, learningRate);

    Optimize::SGD(Br, d.Br, learningRate);
    Optimize::SGD(Bz, d.Bz, learningRate);
    Optimize::SGD(Bg, d.Bg, learningRate);
    d.zero();
    return;
}

void RL::GRU::RMSProp(float learningRate, float rho)
{
    Optimize::RMSProp(W, s.W, d.W, learningRate, rho);
    Optimize::RMSProp(B, s.B, d.B, learningRate, rho);

    Optimize::RMSProp(Wr, s.Wr, d.Wr, learningRate, rho);
    Optimize::RMSProp(Wz, s.Wz, d.Wz, learningRate, rho);
    Optimize::RMSProp(Wg, s.Wg, d.Wg, learningRate, rho);

    Optimize::RMSProp(Ur, s.Ur, d.Ur, learningRate, rho);
    Optimize::RMSProp(Uz, s.Uz, d.Uz, learningRate, rho);
    Optimize::RMSProp(Ug, s.Ug, d.Ug, learningRate, rho);

    Optimize::RMSProp(Br, s.Br, d.Br, learningRate, rho);
    Optimize::RMSProp(Bz, s.Bz, d.Bz, learningRate, rho);
    Optimize::RMSProp(Bg, s.Bg, d.Bg, learningRate, rho);

    d.zero();
    return;
}

void RL::GRU::Adam(float learningRate,  float alpha, float beta)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimize::Adam(W, s.W, v.W, d.W, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(B, s.B, v.B, d.B, alpha_t, beta_t, learningRate, alpha, beta);

    Optimize::Adam(Wr, s.Wr, v.Wr, d.Wr, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Wz, s.Wz, v.Wz, d.Wz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Wg, s.Wg, v.Wg, d.Wg, alpha_t, beta_t, learningRate, alpha, beta);

    Optimize::Adam(Ur, s.Ur, v.Ur, d.Ur, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Uz, s.Uz, v.Uz, d.Uz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Ug, s.Ug, v.Ug, d.Ug, alpha_t, beta_t, learningRate, alpha, beta);

    Optimize::Adam(Br, s.Br, v.Br, d.Br, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Bz, s.Bz, v.Bz, d.Bz, alpha_t, beta_t, learningRate, alpha, beta);
    Optimize::Adam(Bg, s.Bg, v.Bg, d.Bg, alpha_t, beta_t, learningRate, alpha, beta);
    d.zero();
    return;
}

void RL::GRU::test()
{
    GRU gru(2, 8, 1, true);
    auto zeta = [](float x, float y) -> float {
        return x*x + y*y + x*y;
    };
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Tensor> data;
    std::vector<Tensor> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Tensor q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Tensor> &batchData,
            std::vector<Tensor> &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = selectIndex(Random::engine);
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Tensor> batchData;
        std::vector<Tensor> batchTarget;
        sample(batchData, batchTarget, 32);
        gru.forward(batchData);
        gru.gradient(batchData, batchTarget);
        gru.Adam(0.001);
    }

    gru.clear();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            auto s = gru.forward(p);
            std::cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
