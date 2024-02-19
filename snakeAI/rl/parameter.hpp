#ifndef PARAMETER_HPP
#define PARAMETER_HPP
#include "mat.hpp"
#include "optimizer.h"

namespace RL {

class GradValue
{
public:
    Mat val;
    Mat d;
    Mat s;
public:
    GradValue(){}
    explicit GradValue(int rows, int cols)
        :val(rows, cols),d(rows, cols),s(rows, cols){}
    GradValue(const GradValue &r):val(r.val),d(r.d),s(r.s){}
    inline float operator[](int index) const {return val.val[index];}
    inline float& operator[](int index) {return val.val[index];}
    void RMSProp(float rho, float learningRate, float decay)
    {
        Optimizer::RMSProp(val, s, d, learningRate, rho, decay);
        d.zero();
        return;
    }

};

}
#endif // PARAMETER_HPP
