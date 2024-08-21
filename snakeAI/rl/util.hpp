#ifndef UTIL_HPP
#define UTIL_HPP
#include <vector>
#include <random>
#include <ctime>
#include "tensor.hpp"

namespace RL {

constexpr static float pi = 3.1415926535898;

struct Random {
    static std::default_random_engine engine;
    static std::random_device device;
    static std::mt19937 generator;

    inline static int categorical(const Tensor& p)
    {
        std::discrete_distribution<int> distribution(p.begin(), p.end());
        return distribution(Random::generator);
    }
    inline static void uniform(Tensor &x, float x1, float x2)
    {
        std::uniform_real_distribution<float> distribution(x1, x2);
        for (std::size_t i = 0; i < x.size(); i++) {
            x[i] = distribution(Random::engine);
        }
        return;
    }
    inline static void bernoulli(Tensor &x, float p)
    {
        std::bernoulli_distribution distribution(p);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::engine) / (1 - p);
        }
        return;
    }
    inline static void normal(Tensor &x, float u, float sigma)
    {
        std::normal_distribution<float> distribution(u, sigma);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::generator);
        }
        return;
    }
};

namespace Norm {
    inline float l1(const Tensor &x1, const Tensor& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = std::abs(x1[i] - x2[i]);
            s += d;
        }
        return s;
    }
    inline float l2(const Tensor &x1, const Tensor& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += d*d;
        }
        return std::sqrt(s);
    }
    inline float lp(const Tensor &x1, const Tensor& x2, float p)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += std::pow(d, p);
        }
        return std::pow(s, 1.0/p);
    }
    inline float l8(const Tensor &x1, const Tensor& x2)
    {
        float s = x1[0] - x2[0];
        for (std::size_t i = 1; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            if (d > s) {
                s = d;
            }
        }
        return s;
    }
}

inline Tensor& sqrt(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sqrt(x[i]);
    }
    return x;
}

inline Tensor& exp(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i]);
    }
    return x;
}

inline Tensor& log(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::log(x[i]);
    }
    return x;
}

inline Tensor& tanh(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::tanh(x[i]);
    }
    return x;
}

inline Tensor& sin(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sin(x[i]);
    }
    return x;
}

inline Tensor& cos(Tensor& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::cos(x[i]);
    }
    return x;
}

inline Tensor sqrt(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sqrt(x[i]);
    }
    return x;
}

inline Tensor exp(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::exp(x[i]);
    }
    return y;
}

inline Tensor log(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::log(x[i]);
    }
    return y;
}

inline Tensor tanh(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::tanh(x[i]);
    }
    return y;
}

inline Tensor sin(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sin(x[i]);
    }
    return y;
}

inline Tensor cos(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] /= std::cos(x[i]);
    }
    return y;
}

inline Tensor upTriangle(int rows, int cols)
{
    Tensor x(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < cols; j++) {
            x(i, j) = 1;
        }
    }
    return x;
}

inline Tensor lowTriangle(int rows, int cols)
{
    Tensor x(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < i; j++) {
            x(i, j) = 1;
        }
    }
    return x;
}

/* exponential moving average */
inline void lerp(Tensor &s, const Tensor s_, float r)
{
    for (std::size_t i = 0; i < s.size(); i++) {
        s[i] = (1 - r) * s[i] + r * s_[i];
    }
    return;
}
float gaussian(float x, float u, float sigma);
float clip(float x, float sup, float inf);
float hmean(const Tensor &x);
float gmean(const Tensor &x);
float variance(const Tensor &x, float u);
float covariance(const Tensor& x1, const Tensor& x2);
void zscore(Tensor &x);
void normalize(Tensor &x);


inline float M3(const Tensor &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d;
    }
    return s/float(x.size());
}

inline float M4(const Tensor &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d*d;
    }
    return s/float(x.size());
}

inline void clamp(Tensor &x, float x1, float x2)
{
    std::uniform_real_distribution<float> uniform(x1, x2);
    for (std::size_t i = 0; i < x.size(); i++) {
        float xi = x[i];
        x[i] = xi < x1 ? x1 : xi;
        x[i] = xi > x2 ? x2 : xi;
    }
    return;
}

template<typename T>
inline void uniformRand(T &x, float x1, float x2)
{
    std::uniform_real_distribution<float> uniform(x1, x2);
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = uniform(Random::engine);
    }
    return;
}

inline Tensor& eGreedy(Tensor& x, float exploringRate, bool hard)
{
    std::uniform_real_distribution<float> uniformReal(0, 1);
    float p = uniformReal(Random::engine);
    if (p < exploringRate) {
        if (hard) {
            x.zero();
        }
        std::uniform_int_distribution<int> uniform(0, x.size() - 1);
        int index = uniform(Random::engine);
        x[index] = 1;
    }
    return x;
}

inline Tensor& noise(Tensor& x)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 1);
    x += epsilon;
    x /= x.max();
    return x;
}

inline Tensor& noise(Tensor& x, float exploringRate)
{
    std::uniform_real_distribution<float> uniform(0, 1);
    float p = uniform(Random::engine);
    if (p < exploringRate) {
        Tensor epsilon(x.shape);
        Random::uniform(epsilon, 0, 1);
        x += epsilon;
        x /= x.max();
    }
    return x;
}

inline Tensor& softmax(Tensor &x)
{
    float s = 0;
    float maxValue = x.max();
    x -= maxValue;
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i]);
        s += x[i];
    }
    x /= s;
    return x;
}

inline Tensor& gumbelSoftmax(Tensor &x, float tau)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-8) + 1e-8);
    }
    x += epsilon;
    x /= tau;
    x = softmax(x);
    return x;
}

inline Tensor& gumbelSoftmax(Tensor &x, const Tensor& tau)
{
    Tensor epsilon(x.shape);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-8) + 1e-8);
    }
    x += epsilon;
    x /= tau;
    x = softmax(x);
    return x;
}

inline Tensor& gaussianResample(Tensor &z, float u, float sigma)
{
    /*
        z = u + std*eps
        eps ~ N(0, 1)
    */
    float std = std::sqrt(sigma);
    Tensor eps(z.shape);
    Random::normal(eps, u, std);
    for (std::size_t i = 0; i < z.totalSize; i++) {
        z[i] = u + std*eps[i];
    }
    return z;
}

}
#endif // UTIL_HPP
