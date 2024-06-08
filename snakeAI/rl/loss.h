#ifndef LOSS_H
#define LOSS_H
#include <iostream>
#include "tensor.hpp"

namespace RL {

/* loss type */
namespace Loss {

inline Tensor MSE(const Tensor& yo, const Tensor& yt)
{
    Tensor loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
         loss[i] = 2*(yo[i] - yt[i]);
    }
    return loss;
}
inline Tensor CrossEntropy(const Tensor& yo, const Tensor& yt)
{
    Tensor loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
        loss[i] = -yt[i]*std::log(yo[i] + 1e-8);
        //loss[i] = -yt[i]/(yo[i] + 1e-8);
    }
    return loss;
}
inline Tensor BCE(const Tensor& yo, const Tensor& yt)
{
    Tensor loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
        loss[i] = -(yt[i]*std::log(yo[i]) + (1 - yt[i])*std::log(1 - yo[i]));
    }
    return loss;
}

};

}
#endif // LOSS_H
