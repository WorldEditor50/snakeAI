#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "tensor.hpp"
#include "util.hpp"

namespace RL {

namespace Optimize {

inline void SGD(Tensor &w, Tensor &dw,
                float lr, float decay=0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        w[i] = (1 - decay)*w[i] - lr*dw[i];
    }
    return;
}

inline void SGDM(Tensor &w, Tensor &m, Tensor &dw,
                 float lr, float alpha,
                 float decay=0, float clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = (1 - decay)*m[i] - dw[i]*alpha;
        w[i] -= lr * m[i];
    }
    return;
}

inline void Adagrad(Tensor &w, Tensor &r, Tensor &dw,
                    float lr, float decay=0, float clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        r[i] += dw[i]*dw[i];
        w[i] = (1 - decay)*w[i] - lr*dw[i]/(std::sqrt(r[i]) + 1e-9);
    }
    return;
}

inline void AdaDelta(Tensor &w, Tensor &v, Tensor &delta, Tensor &dwPrime, Tensor &dw,
                     float lr, float rho, float decay = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho * v[i] + (1 - rho) * dw[i] * dw[i];
        delta[i] = rho*delta[i] + (1 - rho)*dwPrime[i]*dwPrime[i];
        dwPrime[i] = std::sqrt((delta[i] + 1e-9)/(v[i] + 1e-9));
        w[i] = (1 - decay)*w[i] - lr*dwPrime[i];
    }
    return;
}

inline void RMSProp(Tensor &w, Tensor &v, Tensor &dw,
                    float lr, float rho=0.9,
                    float decay = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho*v[i] + (1 - rho)*dw[i]*dw[i];
        w[i] = (1 - decay)*w[i] - lr*dw[i]/(std::sqrt(v[i]) + 1e-9);
    }
    return;
}

inline void Adam(Tensor &w, Tensor &v, Tensor &m, Tensor &dw,
                 float alpha_, float beta_,
                 float lr, float alpha, float beta,
                 float decay = 0, bool clipGrad=true)
{
    if (clipGrad) {
        dw /= dw.norm2();
    }
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = alpha * m[i] + (1 - alpha) * dw[i];
        v[i] = beta * v[i] + (1 - beta) * dw[i] * dw[i];
        float m_ = m[i] / (1 - alpha_);
        float v_ = v[i] / (1 - beta_);
        w[i] = (1 - decay)*w[i] - lr*m_/(std::sqrt(v_) + 1e-9);
    }
    return;
}

inline void clamp(Tensor &w, float c0, float cn)
{
    std::uniform_real_distribution<float> uniform(c0, cn);
    for (std::size_t i = 0; i < w.totalSize; i++) {
        if (w[i] > cn || w[i] < c0) {
            w[i] = uniform(RL::Random::engine);
        }
    }
    return;
}

} // optimizer
} // RL
#endif // OPTIMIZER_H
