#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "util.h"
#include "mat.hpp"

namespace RL {

namespace Optimize {

inline void SGD(Mat &w, const Mat &dw, float learningRate)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        w[i] -= learningRate * dw[i];
    }
    return;
}

inline void SGDM(Mat &w, Mat &m, const Mat &dw, float learningRate, float alpha, float decay=0)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = (1 - decay)*m[i] - dw[i]*alpha;
        w[i] -= learningRate * m[i];
    }
    return;
}

inline void Adagrad(Mat &w, Mat &r, const Mat &dw, float learningRate)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        r[i] += dw[i]*dw[i];
        w[i] -= learningRate*dw[i]/(std::sqrt(r[i]) + 1e-9);
    }
    return;
}

inline void AdaDelta(Mat &w, Mat &v, Mat &delta, Mat &dwPrime, const Mat &dw, float learningRate, float rho)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho * v[i] + (1 - rho) * dw[i] * dw[i];
        delta[i] = rho*delta[i] + (1 - rho)*dwPrime[i]*dwPrime[i];
        dwPrime[i] = std::sqrt((delta[i] + 1e-9)/(v[i] + 1e-9));
        w[i] -= learningRate*dwPrime[i];
    }
    return;
}

inline void RMSProp(Mat &w, Mat &v, const Mat &dw, float learningRate, float rho, float decay = 0)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        v[i] = rho * v[i] + (1 - rho) * dw[i] * dw[i];
        w[i] = (1 - decay)*w[i] - learningRate * dw[i] / (std::sqrt(v[i]) + 1e-9);
    }
    return;
}

inline void Adam(Mat &w, Mat &v, Mat &m, const Mat &dw, float alpha_, float beta_, float learningRate, float alpha, float beta, float decay = 0)
{
    for (std::size_t i = 0; i < w.totalSize; i++) {
        m[i] = alpha * m[i] + (1 - alpha) * dw[i];
        v[i] = beta * v[i] + (1 - beta) * dw[i] * dw[i];
        float m_ = m[i] / (1 - alpha_);
        float v_ = v[i] / (1 - beta_);
        w[i] = (1 - decay)*w[i] - learningRate * m_ / (std::sqrt(v_) + 1e-9);
    }
    return;
}

inline void clamp(Mat &w, float c0, float cn)
{
    std::uniform_real_distribution<float> uniform(c0, cn);
    for (std::size_t i = 0; i < w.totalSize; i++) {
        if (w[i] > cn || w[i] < c0) {
            w[i] = uniform(RL::Rand::engine);
        }
    }
    return;
}

} // optimizer
} // RL
#endif // OPTIMIZER_H
