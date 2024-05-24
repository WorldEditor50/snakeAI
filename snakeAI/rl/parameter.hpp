#ifndef PARAMETER_HPP
#define PARAMETER_HPP
#include "mat.hpp"
#include "optimize.h"

namespace RL {

class GradValue
{
public:
    Mat val;
    Mat g;
    Mat v;
public:
    GradValue(){}
    explicit GradValue(int rows, int cols)
        :val(rows, cols),g(rows, cols),v(rows, cols){}
    GradValue(const GradValue &r):val(r.val),g(r.g),v(r.v){}
    inline float operator[](int index) const {return val.val[index];}
    inline float& operator[](int index) {return val.val[index];}
    void RMSProp(float rho, float learningRate, float decay)
    {
        Optimize::NormRMSProp(val, v, g, learningRate, rho, decay);
        for (std::size_t i = 0; i < val.totalSize; i++) {
            val[i] = val[i] < 1e-3 ? 1e-3 : val[i];
        }
        g.zero();
        return;
    }

};


}
#endif // PARAMETER_HPP
