#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <random>
#include <ctime>
#include "mat.hpp"

namespace RL {

constexpr static float pi = 3.1415926535898;

struct Rand {
    static std::default_random_engine engine;
};

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
float variance(const Mat &x);
float variance(const Mat &x, float u);
float covariance(const Mat& x1, const Mat& x2);
void zscore(Mat &x);
void normalize(Mat &x);

template<typename T>
inline void uniformRand(T &x, float x1, float x2)
{
    std::uniform_real_distribution<float> uniform(x1, x2);
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = uniform(Rand::engine);
    }
    return;
}

}
#endif // UTIL_H
