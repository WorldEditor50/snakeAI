#ifndef LOSS_H
#define LOSS_H
#include <iostream>
#include "tensor.hpp"

namespace RL {

/* loss type */
namespace Loss {

struct MSE {
    inline static Tensor f(const Tensor& yo, const Tensor& yt)
    {
        Tensor loss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
            float d = yo[i] - yt[i];
            loss[i] = 2*d*d;
        }
        return loss;
    }

    inline static Tensor df(const Tensor& yo, const Tensor& yt)
    {
        Tensor dLoss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
             dLoss[i] = 2*(yo[i] - yt[i]);
        }
        return dLoss;
    }
};

struct CrossEntropy {
    inline static Tensor f(const Tensor& yo, const Tensor& yt)
    {
        Tensor loss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
            loss[i] = -yt[i]*std::log(yo[i]);
        }
        return loss;
    }

    inline static Tensor df(const Tensor& yo, const Tensor& yt)
    {
        Tensor dLoss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
             dLoss[i] = -yt[i]/(yo[i]);
        }
        return dLoss;
    }

};

struct BCE {
    inline static Tensor f(const Tensor& yo, const Tensor& yt)
    {
        Tensor loss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
            loss[i] = -(yt[i]*std::log(yo[i]) + (1 - yt[i])*std::log(1 - yo[i]));
        }
        return loss;
    }
    inline static Tensor df(const Tensor& yo, const Tensor& yt)
    {
        Tensor dLoss(yo.totalSize, 1);
        for (std::size_t i = 0; i < yo.size(); i++) {
             dLoss[i] = -yt[i]/(yo[i]) - (1 - yt[i])/(1 - yo[i]);
        }
        return dLoss;
    }
};

} //Loss
} //RL
#endif // LOSS_H
