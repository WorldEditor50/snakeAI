#ifndef LOSS_H
#define LOSS_H
#include <iostream>
#include "mat.hpp"

namespace RL {

/* loss type */
namespace Loss {

inline Mat MSE(const Mat& yo, const Mat& yt)
{
    Mat loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
         loss[i] = 2*(yo[i] - yt[i]);
    }
    return loss;
}
inline Mat CrossEntropy(const Mat& yo, const Mat& yt)
{
    Mat loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
        loss[i] = -yt[i]*std::log(yo[i] + 1e-9);
    }
    return loss;
}
inline Mat BCE(const Mat& yo, const Mat& yt)
{
    Mat loss(yo.totalSize, 1);
    for (std::size_t i = 0; i < yo.size(); i++) {
        loss[i] = -(yt[i]*std::log(yo[i]) + (1 - yt[i])*std::log(1 - yo[i]));
    }
    return loss;
}

};

}
#endif // LOSS_H
