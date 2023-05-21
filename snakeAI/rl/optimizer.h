#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "util.h"

namespace RL {

namespace  Optimizer {

inline void SGD(Mat &w, const Mat &dw, float learningRate)
{
    for (std::size_t i = 0; i < w.size(); i++) {
        w[i] -= learningRate * dw[i];
    }
    return;
}

inline void Adagrad(Mat &w, Mat &r, const Mat &dw, float learningRate)
{
    for (std::size_t i = 0; i < w.size(); i++) {
        r[i] += dw[i]*dw[i];
        w[i] -= learningRate*dw[i]/(std::sqrt(r[i]) + 1e-9);
    }
    return;
}

inline void AdaDelta(Mat &w, Mat &S, Mat &delta, Mat &dwPrime, const Mat &dw, float learningRate, float rho)
{
    for (std::size_t i = 0; i < w.size(); i++) {
        S[i] = rho * S[i] + (1 - rho) * dw[i] * dw[i];
        delta[i] = rho*delta[i] + (1 - rho)*dwPrime[i]*dwPrime[i];
        dwPrime[i] = std::sqrt((delta[i] + 1e-9)/(S[i] + 1e-9));
        w[i] -= learningRate*dwPrime[i];
    }
    return;
}

inline void RMSProp(Mat &w, Mat &S, const Mat &dw, float learningRate, float rho, float decay = 0.01)
{
    for (std::size_t i = 0; i < w.size(); i++) {
        S[i] = rho * S[i] + (1 - rho) * dw[i] * dw[i];
        w[i] = (1 - decay)*w[i] - learningRate * dw[i] / (std::sqrt(S[i]) + 1e-9);
    }
    return;
}

inline void Adam(Mat &w, Mat &S, Mat &V, const Mat &dw, float alpha_, float beta_, float learningRate, float alpha, float beta, float decay = 0.01)
{
    for (std::size_t i = 0; i < w.size(); i++) {
        V[i] = alpha * V[i] + (1 - alpha) * dw[i];
        S[i] = beta * S[i] + (1 - beta) * dw[i] * dw[i];
        float v = V[i] / (1 - alpha_);
        float s = S[i] / (1 - beta_);
        w[i] = (1 - decay)*w[i] - learningRate * v / (std::sqrt(s) + 1e-9);
    }
    return;
}

inline void clamp(Mat &w, float c0, float cn)
{
    std::uniform_real_distribution<float> distribution(c0, cn);
    for (std::size_t i = 0; i < w.size(); i++) {
        if (w[i] > cn || w[i] < c0) {
            w[i] = distribution(RL::Rand::engine);
        }
    }
    return;
}

} // optimizer
} // RL
#endif // OPTIMIZER_H
