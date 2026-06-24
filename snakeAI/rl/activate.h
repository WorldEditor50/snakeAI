#ifndef ACTIVATE_H
#define ACTIVATE_H
#include <cmath>
#include "tensor.hpp"

namespace RL {

struct Sigmoid {
    inline static float f(float x) {return 1.0/(1 + std::exp(-1.702*x));}
    inline static float df(float y) {return 1.702*y*(1 - y);}
};

struct Tanh {
    inline static float f(float x) {return std::tanh(x);}
    inline static float df(float y) {return 1 - y*y;}
};
struct Softplus {
    inline static float f(float x) {return std::log(1 + std::exp(x));}
    inline static float df(float y) {return 1 - std::exp(-y);}
};
struct Relu {
    inline static float f(float x) {return x > 0 ? x : 0;}
    inline static float df(float y) {return y > 0 ? 1 : 0;}
};

struct LeakyRelu {
    inline static float f(float x) {return x > 0 ? x : 0.01*x;}
    inline static float df(float y) {return y > 0 ? 1 : 0.01;}
};

struct Selu {
    inline static float f(float x)
    {
        float y = x;
        if (y > 1) {
            y = 1;
        } else if (y < -1) {
            y = -1;
        }
        return y;
    }
    inline static float df(float y)
    {
        float dy = 0;
        if (y >= -1 && y <= 1) {
            dy = 1;
        }
        return dy;
    }
};

struct Linear {
    inline static float f(float x) {return x;}
    inline static float df(float) {return 1;}
};

struct Mish {
    inline static float f(float x) {return x*std::tanh(std::log(1 + std::exp(x)));}
    inline static float df(float x)
    {
        float y0 = std::exp(x);
        float y1 = std::tanh(std::log(1 + y0));
        float y2 = x*(1 - y1*y1)*y0/(1 + y0);
        return y1 + y2;
    }
};

struct XTanh {
    inline static float f(float x) {return std::tanh(x*std::log(1 + std::exp(x)));}
    inline static float df(float x, float y)
    {
        float y0 = std::exp(x);
        float y1 = std::log(1 + y0);
        float y2 = (1 - y*y)*(y1 + x*y0/(1 + y0));
        return y2;
    }
};

struct Swish {
    static constexpr float beta = 1.702;
    inline static float f(float x) {return x/(1 + std::exp(-beta*x));}
    inline static float df(float x)
    {
        float s = 1.0/(1 + std::exp(-beta*x));
        return s + beta*x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr float c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr float c2 = 0.044715;
    inline static float f(float x)
    {
        return 0.5*x*(1 + std::tanh(c1*(x + c2*x*x*x)));
    }
    inline static float df(float x)
    {
        float t = std::tanh(c1*(x + c2*x*x*x));
        return 0.5*(1 + t + x*(c1*(1 + 3*c2*x*x)*(1 - t*t)));
    }
};

struct Softmax {
    inline static Tensor f(const Tensor &x)
    {
        Tensor y = x;
        float s = 0;
        float maxValue = x.max();
        y -= maxValue;
        for (std::size_t i = 0; i < y.size(); i++) {
            y[i] = std::exp(y[i]);
            s += y[i];
        }
        y /= s;
        return y;
    }
    inline static Tensor jacobian(const Tensor &y)
    {
        Tensor J(y.totalSize, y.totalSize);
        for (std::size_t i = 0; i < y.totalSize; i++) {
            for (std::size_t j = 0; j < y.totalSize; j++) {
                if (i == j) {
                    J(i, j) = y[i]*(1 - y[j]);
                } else {
                    J(i, j) = -y[i]*y[j];
                }
            }
        }
        return J;
    }

    /*
        O(N) softmax Jacobian-vector product: compute v_out = J^T · v_in
        WITHOUT constructing the O(N²) Jacobian matrix.

        For softmax output y = σ(x) (tensor of arbitrary shape):
            J[i,j] = y[i]·(δ_ij - y[j])  (symmetric: J = J^T, treats y as flat)

        Therefore:
            (J^T · v)[i] = y[i] · (v[i] - Σ_k y[k]·v[k])

        Time: O(N), Memory: O(1)
        vs constructing J explicitly: O(N²) time, O(N²) memory

        This works for any tensor shape — y and v are treated as flat vectors.
        When y is a 2D matrix (as in ScaledDotProduct's N×N attention matrix z),
        all N² elements are treated uniformly (matching the flat softmax behavior
        used in RL::softmax() and Softmax::f()).

        Parameters:
            y:  softmax output (any shape, treated as flat in the Jacobian)
            v:  gradient w.r.t. y (same shape as y)
            out: result = J^T · v (same shape as y), can alias v
    */
    inline static void jacobian_transpose_mul(const Tensor &y, const Tensor &v, Tensor &out)
    {
        /* dot = Σ_k y[k] · v[k]  (single scalar over all elements) */
        float dot = 0;
        for (std::size_t k = 0; k < y.totalSize; k++) {
            dot += y[k] * v[k];
        }
        /* out[i] = y[i] · (v[i] - dot) */
        for (std::size_t i = 0; i < y.totalSize; i++) {
            out[i] = y[i] * (v[i] - dot);
        }
        return;
    }
};

struct Zeta {
    inline static Tensor f(Tensor &x, float &gamma)
    {
        /*
            u = 1/n∑x
            sigma = 1/n ∑(x - u)*(x - u)
        */
        float u = x.mean();
        float sigma = x.variance(u);
        gamma = 1.0/std::sqrt(sigma);
        Tensor x_(x.shape);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x_[i] = (x[i] - u)*gamma;
        }
        return x_;
    }

    inline static Tensor jacobian(Tensor &x)
    {
        Tensor J(x.totalSize, x.totalSize);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            for (std::size_t j = 0; j < x.totalSize; j++) {
                if (i == j) {

                } else {

                }
            }
        }
        return J;
    }
};

}
#endif // ACTIVATE_H
