#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "util.h"

namespace RL {

struct Optimizer
{
    static void SGD(Mat &W, const Mat &dW, double learningRate)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] -= learningRate * dW[i][j];
            }
        }
        return;
    }
    static void SGD(Vec &B, const Vec &dB, double learningRate)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            B[i] -= learningRate * dB[i];
        }
        return;
    }
    static void RMSProp(Mat &W, Mat &Sw, const Mat &dW, double learningRate, double rho)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
                W[i][j] -= learningRate * dW[i][j] / (sqrt(Sw[i][j]) + 1e-9);
            }
        }
        return;
    }
    static void RMSProp(Vec &B, Vec &Sb, const Vec &dB, double learningRate, double rho)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
            B[i] -= learningRate * dB[i] / (sqrt(Sb[i]) + 1e-9);
        }
        return;
    }
    static void Adam(Mat &W, Mat &Sw, Mat &Vw, const Mat &dW, double alpha_, double beta_, double learningRate, double alpha, double beta)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                Vw[i][j] = alpha *Vw[i][j] + (1 - alpha) * dW[i][j];
                Sw[i][j] = beta * Sw[i][j] + (1 - beta) * dW[i][j] * dW[i][j];
                double v = Vw[i][j] / (1 - alpha_);
                double s = Sw[i][j] / (1 - beta_);
                W[i][j] -= learningRate * v / (sqrt(s) + 1e-9);
            }
        }
        return;
    }
    static void Adam(Vec &B, Vec &Sb, Vec &Vb, const Vec &dB, double alpha_, double beta_, double learningRate, double alpha, double beta)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            Vb[i] = alpha * Vb[i] + (1 - alpha) * dB[i];
            Sb[i] = beta * Sb[i] + (1 - beta) * dB[i] * dB[i];
            double v = Vb[i] / (1 - alpha_);
            double s = Sb[i] / (1 - beta_);
            B[i] -= learningRate * v / (sqrt(s) + 1e-9);
        }
        return;
    }
};

}
#endif // OPTIMIZER_H
