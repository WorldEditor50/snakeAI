#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include "util.h"

namespace RL {

/* activate method */
struct Sigmoid {
    inline static double _(double x) {return 1.0/(1 + exp(-x));}
    inline static double d(double y) {return y*(1 - y);}
};

struct Tanh {
    inline static double _(double x) {return tanh(x);}
    inline static double d(double y) {return 1 - y*y;}
};

struct Relu {
    inline static double _(double x) {return x > 0 ? x : 0;}
    inline static double d(double y) {return y > 0 ? 1 : 0;}
};

struct LeakyRelu {
    inline static double _(double x) {return x > 0 ? x : 0.01*x;}
    inline static double d(double y) {return y > 0 ? 1 : 0.01;}
};

struct Linear {
    inline static double _(double x) {return x;}
    inline static double d(double) {return 1;}
};

struct Swish {
    static constexpr double beta = 1.0;//1.702;
    inline static double _(double x) {return x*Sigmoid::_(beta*x);}
    inline static double d(double x)
    {
        double s = Sigmoid::_(beta*x);
        return s + x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr double c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr double c2 = 0.044715;
    inline static double _(double x)
    {
        return 0.5*x*(1 + tanh(c1*(x + c2*x*x*x)));
    }
    inline static double d(double x)
    {
        double t = tanh(c1*(x + c2*x*x*x));
        return 0.5*(1 + t + x*(c1*(1 + 3*c2*x*x)*(1 - t*t)));
    }
};

}
#endif // ACTIVATE_H
