#include "util.h"

std::default_random_engine RL::Rand::engine(std::random_device{}());

void RL::zscore(Mat &x)
{
    /* sigma */
    float sigma = variance(x);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] /= sigma;
    }
    return;
}

void RL::normalize(Mat &x)
{
    float minValue = x[0];
    float maxValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (minValue > x[i]) {
            minValue = x[i];
        }
        if (maxValue < x[i]) {
            maxValue = x[i];
        }
    }
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = (x[i] - minValue) / (maxValue - minValue);
    }
    return;
}

float RL::variance(const Mat &x)
{
    /* expectation */
    float u = x.mean();
    float sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sqrt(sigma / x.size());
}

float RL::variance(const Mat &x, float u)
{
    float sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sqrt(sigma / x.size());
}

float RL::covariance(const Mat &x1, const Mat &x2)
{
    float u = x1.mean();
    float v = x2.mean();
    float covar = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
         covar += (x1[i] - u) * (x2[i] - v);
    }
    return covar;
}

float RL::clip(float x, float sup, float inf)
{
    float y = x;
    if (x < sup) {
        y = sup;
    } else if (x > inf) {
        y = inf;
    }
    return y;
}

float RL::hmean(const RL::Mat &x)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        s += 1/x[i];
    }
    return float(x.size())/s;
}

float RL::gmean(const RL::Mat &x)
{
    float s = 1;
    for (std::size_t i = 0; i < x.size(); i++) {
        s *= x[i];
    }
    return pow(s, 1.0/x.size());
}

float RL::gaussian(float x, float u, float sigma)
{
    return 1/sqrt(2*pi*sigma)*exp(-0.5*(x - u)*(x - u)/sigma);
}
