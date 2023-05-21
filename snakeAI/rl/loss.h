#ifndef LOSS_H
#define LOSS_H
#include "util.h"

namespace RL {

/* loss type */
struct Loss
{
    static void MSE(Mat& E, const Mat& O, const Mat& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
             E[i] = 2*(O[i] - y[i]);
        }
        return;
    }
    static void CROSS_EMTROPY(Mat& E, const Mat& O, const Mat& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
            E[i] = -y[i] * log(O[i]);
        }
        return;
    }
    static void BCE(Mat& E, const Mat& O, const Mat& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
            E[i] = -(y[i] * log(O[i]) + (1 - y[i]) * log(1 - O[i]));
        }
        return;
    }
};

}
#endif // LOSS_H
