#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include "util.h"

namespace RL {

/* activate method */
struct Sigmoid {
    inline static float _(float x) {return 1.0/(1 + std::exp(-x));}
    inline static float d(float y) {return y*(1 - y);}
};

struct Tanh {
    inline static float _(float x) {return std::tanh(x);}
    inline static float d(float y) {return 1 - y*y;}
};

struct Relu {
    inline static float _(float x) {return x > 0 ? x : 0;}
    inline static float d(float y) {return y > 0 ? 1 : 0;}
};

struct LeakyRelu {
    inline static float _(float x) {return x > 0 ? x : 0.01*x;}
    inline static float d(float y) {return y > 0 ? 1 : 0.01;}
};

struct Linear {
    inline static float _(float x) {return x;}
    inline static float d(float) {return 1;}
};

struct Swish {
    static constexpr float beta = 1.0;//1.702;
    inline static float _(float x) {return x*Sigmoid::_(beta*x);}
    inline static float d(float x)
    {
        float s = Sigmoid::_(beta*x);
        return s + x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr float c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr float c2 = 0.044715;
    inline static float _(float x)
    {
        return 0.5*x*(1 + std::tanh(c1*(x + c2*x*x*x)));
    }
    inline static float d(float x)
    {
        float t = std::tanh(c1*(x + c2*x*x*x));
        return 0.5*(1 + t + x*(c1*(1 + 3*c2*x*x)*(1 - t*t)));
    }
};

}
#endif // ACTIVATE_H
