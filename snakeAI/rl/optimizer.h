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
    static void Adagrad(Mat &W, Mat &r, const Mat &dW, double learningRate)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                r[i][j] += dW[i][j]*dW[i][j];
                W[i][j] -= learningRate*dW[i][j]/(sqrt(r[i][j]) + 1e-9);
            }
        }
        return;
    }
    static void Adagrad(Vec &B, Vec &r, const Vec &dB, double learningRate)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            r[i] += dB[i]*dB[i];
            B[i] -= learningRate*dB[i]/(sqrt(r[i]) + 1e-9);
        }
        return;
    }
    static void AdaDelta(Mat &W, Mat &Sw, Mat &delta, Mat &dWPrime, const Mat &dW, double learningRate, double rho)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                Sw[i][j] = rho*Sw[i][j] + (1 - rho)*dW[i][j]*dW[i][j];
                delta[i][j] = rho*delta[i][j] + (1 - rho)*dWPrime[i][j]*dWPrime[i][j];
                dWPrime[i][j] = sqrt((delta[i][j] + 1e-9)/(Sw[i][j] + 1e-9));
                W[i][j] -= learningRate*dWPrime[i][j];
            }
        }
        return;
    }
    static void AdaDelta(Vec &B, Vec &Sb, Vec &delta, Vec &dBPrime, const Vec &dB, double learningRate, double rho)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
            delta[i] = rho*delta[i] + (1 - rho)*dBPrime[i]*dBPrime[i];
            dBPrime[i] = sqrt((delta[i] + 1e-9)/(Sb[i] + 1e-9));
            B[i] -= learningRate*dBPrime[i];
        }
        return;
    }
    static void RMSProp(Mat &W, Mat &Sw, const Mat &dW, double learningRate, double rho, double decay = 0.01)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                Sw[i][j] = rho*Sw[i][j] + (1 - rho)*dW[i][j]*dW[i][j];
                W[i][j] = (1 - decay)*W[i][j] - learningRate*dW[i][j]/(sqrt(Sw[i][j]) + 1e-9);
            }
        }
        return;
    }
    static void RMSProp(Vec &B, Vec &Sb, const Vec &dB, double learningRate, double rho, double decay = 0.01)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
            B[i] = (1 - decay)*B[i] - learningRate * dB[i] / (sqrt(Sb[i]) + 1e-9);
        }
        return;
    }
    static void Adam(Mat &W, Mat &Sw, Mat &Vw, const Mat &dW, double alpha_, double beta_, double learningRate, double alpha, double beta, double decay=0.01)
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                Vw[i][j] = alpha *Vw[i][j] + (1 - alpha) * dW[i][j];
                Sw[i][j] = beta * Sw[i][j] + (1 - beta) * dW[i][j] * dW[i][j];
                double v = Vw[i][j] / (1 - alpha_);
                double s = Sw[i][j] / (1 - beta_);
                W[i][j] = (1 - decay)*W[i][j] - learningRate * v / (sqrt(s) + 1e-9);
            }
        }
        return;
    }
    static void Adam(Vec &B, Vec &Sb, Vec &Vb, const Vec &dB, double alpha_, double beta_, double learningRate, double alpha, double beta, double decay = 0.01)
    {
        for (std::size_t i = 0; i < B.size(); i++) {
            Vb[i] = alpha * Vb[i] + (1 - alpha) * dB[i];
            Sb[i] = beta * Sb[i] + (1 - beta) * dB[i] * dB[i];
            double v = Vb[i] / (1 - alpha_);
            double s = Sb[i] / (1 - beta_);
            B[i] = (1 - decay)*B[i] - learningRate * v / (sqrt(s) + 1e-9);
        }
        return;
    }
    static void clamp(Mat &W, double c0, double cn)
    {
        std::uniform_real_distribution<double> distribution(c0, cn);
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                if (W[i][j] > cn || W[i][j] < c0) {
                    W[i][j] = distribution(RL::Rand::engine);
                }
            }
        }
        return;
    }
    static void clamp(Vec &B, double c0, double cn)
    {
        std::uniform_real_distribution<double> distribution(c0, cn);
        for (std::size_t i = 0; i < B.size(); i++) {
            if (B[i] > cn || B[i] < c0) {
                B[i] = distribution(RL::Rand::engine);
            }
        }
        return;
    }
};

}
#endif // OPTIMIZER_H
