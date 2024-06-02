#ifndef UTIL_HPP
#define UTIL_HPP
#include <vector>
#include <random>
#include <ctime>
#include "mat.hpp"

namespace RL {

constexpr static float pi = 3.1415926535898;

struct Random {
    static std::default_random_engine engine;
    static std::random_device device;
    static std::mt19937 generator;

    inline static int categorical(const Mat& p)
    {
        std::discrete_distribution<int> distribution(p.begin(), p.end());
        return distribution(Random::generator);
    }
    inline static void uniform(Mat &x, float x1, float x2)
    {
        std::uniform_real_distribution<float> distribution(x1, x2);
        for (std::size_t i = 0; i < x.size(); i++) {
            x[i] = distribution(Random::engine);
        }
        return;
    }
    inline static void bernoulli(Mat &x, float p)
    {
        std::bernoulli_distribution distribution(p);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::engine) / (1 - p);
        }
        return;
    }
    inline static void normal(Mat &x, float u, float sigma)
    {
        std::normal_distribution<float> distribution(u, sigma);
        for (std::size_t i = 0; i <x.size(); i++) {
            x[i] = distribution(Random::generator);
        }
        return;
    }
};

namespace Norm {
    inline float l1(const Mat &x1, const Mat& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = std::abs(x1[i] - x2[i]);
            s += d;
        }
        return s;
    }
    inline float l2(const Mat &x1, const Mat& x2)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += d*d;
        }
        return std::sqrt(s);
    }
    inline float lp(const Mat &x1, const Mat& x2, float p)
    {
        float s = 0;
        for (std::size_t i = 0; i < x1.size(); i++) {
            float d = x1[i] - x2[i];
            s += std::pow(d, p);
        }
        return std::pow(s, 1.0/p);
    }
    inline float l8(const Mat &x1, const Mat& x2)
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

inline Mat& sqrt(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sqrt(x[i]);
    }
    return x;
}

inline Mat& exp(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i]);
    }
    return x;
}

inline Mat& log(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::log(x[i]);
    }
    return x;
}

inline Mat& tanh(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::tanh(x[i]);
    }
    return x;
}

inline Mat& sin(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::sin(x[i]);
    }
    return x;
}

inline Mat& cos(Mat& x)
{
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = std::cos(x[i]);
    }
    return x;
}

inline Mat sqrt(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sqrt(x[i]);
    }
    return x;
}

inline Mat exp(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::exp(x[i]);
    }
    return y;
}

inline Mat log(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::log(x[i]);
    }
    return y;
}

inline Mat tanh(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::tanh(x[i]);
    }
    return y;
}

inline Mat sin(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = std::sin(x[i]);
    }
    return y;
}

inline Mat cos(const Mat& x)
{
    Mat y(x.rows, x.cols);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] /= std::cos(x[i]);
    }
    return y;
}

/* exponential moving average */
inline void EMA(Mat &s, const Mat s_, float r)
{
    for (std::size_t i = 0; i < s.size(); i++) {
        s[i] = (1 - r) * s[i] + r * s_[i];
    }
    return;
}
float gaussian(float x, float u, float sigma);
float clip(float x, float sup, float inf);
float hmean(const Mat &x);
float gmean(const Mat &x);
float variance(const Mat &x, float u);
float covariance(const Mat& x1, const Mat& x2);
void zscore(Mat &x);
void normalize(Mat &x);


inline float M3(const Mat &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d;
    }
    return s/float(x.size());
}

inline float M4(const Mat &x, float u)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        float d = x[i] - u;
        s += d*d*d*d;
    }
    return s/float(x.size());
}

inline void clamp(Mat &x, float x1, float x2)
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

inline Mat& eGreedy(Mat& x, float exploringRate, bool hard)
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

inline Mat& noise(Mat& x)
{
    Mat epsilon(x);
    uniformRand(epsilon, -1, 1);
    x += epsilon;
    float s = x.max();
    x /= s;
    clamp(x, 0, 1);
    return x;
}

inline Mat& noise(Mat& x, float exploringRate)
{
    std::uniform_real_distribution<float> uniform(0, 1);
    float p = uniform(Random::engine);
    if (p < exploringRate) {
        Mat epsilon(x);
        uniformRand(epsilon, -1, 1);
        x += epsilon;
        float s = x.max();
        x /= s;
        clamp(x, 0, 1);
    }
    return x;
}

inline Mat& softmax(Mat &x)
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

inline Mat& gumbelSoftmax(Mat &x, float tau)
{
    Mat epsilon(x);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-8) + 1e-8);
    }
    x += epsilon;
    x /= tau;
    x = softmax(x);
    return x;
}

inline Mat& gumbelSoftmax(Mat &x, const Mat& tau)
{
    Mat epsilon(x);
    Random::uniform(epsilon, 0, 1);
    for (std::size_t i = 0; i < epsilon.size(); i++) {
        epsilon[i] = -std::log(-std::log(epsilon[i] + 1e-8) + 1e-8);
    }
    x += epsilon;
    x /= tau;
    x = softmax(x);
    return x;
}

inline Mat& gaussianResample(Mat &z, float u, float sigma)
{
    /*
        z = u + std*eps
        eps ~ N(0, 1)
    */
    float std = std::sqrt(sigma);
    Mat eps(z.rows, z.cols);
    Random::normal(eps, u, std);
    for (std::size_t i = 0; i < z.totalSize; i++) {
        z[i] = u + std*eps[i];
    }
    return z;
}

}
#endif // UTIL_HPP
