#ifndef LOSS_H
#define LOSS_H
#include "util.h"

namespace RL {

/* loss type */
namespace Loss {

inline void MSE(Mat& E, const Mat& O, const Mat& y)
{
    for (std::size_t i = 0; i < O.size(); i++) {
         E[i] = 2*(O[i] - y[i]);
    }
    return;
}
inline void CrossEntropy(Mat& E, const Mat& O, const Mat& y)
{
    for (std::size_t i = 0; i < O.size(); i++) {
        E[i] = -y[i] * std::log(O[i]);
    }
    return;
}
inline void BCE(Mat& E, const Mat& O, const Mat& y)
{
    for (std::size_t i = 0; i < O.size(); i++) {
        E[i] = -(y[i] * std::log(O[i]) + (1 - y[i]) * std::log(1 - O[i]));
    }
    return;
}

};

}
#endif // LOSS_H
