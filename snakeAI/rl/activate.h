#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include "util.h"
#include "tensor.hpp"

namespace RL {

/* activate method */
struct Sigmoid {
    inline static float f(float x) {return 1.0/(1 + std::exp(-x));}
    inline static float d(float y) {return y*(1 - y);}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = 1.0/(1 + std::exp(-x.val[i]));
        }
        return;
    }
    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i]*(1 - y.val[i]);
        }
        return;
    }
};

struct Tanh {
    inline static float f(float x) {return std::tanh(x);}
    inline static float d(float y) {return 1 - y*y;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = std::tanh(x.val[i]);
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = 1 - y.val[i]*y.val[i];
        }
        return;
    }
};

struct Relu {
    inline static float f(float x) {return x > 0 ? x : 0;}
    inline static float d(float y) {return y > 0 ? 1 : 0;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return;
    }
};

struct LeakyRelu {
    inline static float f(float x) {return x > 0 ? x : 0.01*x;}
    inline static float d(float y) {return y > 0 ? 1 : 0.01;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return;
    }
};

struct Linear {
    inline static float f(float x) {return x;}
    inline static float d(float) {return 1;}
    inline static void f(Tensor &x){}
    inline static void df(Tensor &y)
    {
        y = 1;
        return;
    }
};

struct Swish {
    static constexpr float beta = 1.0;//1.702;
    inline static float f(float x) {return x*Sigmoid::f(beta*x);}
    inline static float d(float x)
    {
        float s = Sigmoid::f(beta*x);
        return s + x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr float c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr float c2 = 0.044715;
    inline static float f(float x)
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
