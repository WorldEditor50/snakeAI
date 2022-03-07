#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <random>
#include <ctime>

namespace RL {

constexpr static double pi = 3.1415926535898;
using T = double;
using Mat = std::vector<std::vector<T> >;
using Vec = std::vector<T>;

struct Rand {
    static std::default_random_engine engine;
};

/* exponential moving average */
inline void EMA(Vec &s, const Vec s_, double r)
{
    for (std::size_t i = 0; i < s.size(); i++) {
        s[i] = (1 - r) * s[i] + r * s_[i];
    }
    return;
}
double gaussian(double x, double u, double sigma);
double clip(double x, double sup, double inf);
int argmax(const Vec &x);
int argmin(const Vec &x);
double max(const Vec &x);
double min(const Vec &x);
double sum(const Vec &x);
double mean(const Vec &x);
double hmean(const Vec &x);
double gmean(const Vec &x);
double variance(const Vec &x);
double variance(const Vec &x, double u);
double covariance(const Vec& x1, const Vec& x2);
void zscore(Vec &x);
void normalize(Vec &x);
double dot(const Vec& x1, const Vec& x2);

}
#endif // UTIL_H
