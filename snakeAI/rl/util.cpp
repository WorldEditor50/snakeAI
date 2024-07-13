#include "util.hpp"

std::random_device RL::Random::device;
std::default_random_engine RL::Random::engine(RL::Random::device());
std::mt19937 RL::Random::generator(RL::Random::device());

void RL::zscore(Tensor &x)
{
    /* sigma */
    float u = x.mean();
    float sigma = std::sqrt(x.variance(u) + 1e-9);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] = (x[i] - u)/sigma;
    }
    return;
}

void RL::normalize(Tensor &x)
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

float RL::variance(const Tensor &x, float u)
{
    float sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sigma / float(x.size());
}

float RL::covariance(const Tensor &x1, const Tensor &x2)
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

float RL::hmean(const RL::Tensor &x)
{
    float s = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        s += 1/x[i];
    }
    return float(x.size())/s;
}

float RL::gmean(const RL::Tensor &x)
{
    float s = 1;
    for (std::size_t i = 0; i < x.size(); i++) {
        s *= x[i];
    }
    return std::pow(s, 1.0/x.size());
}

float RL::gaussian(float x, float u, float sigma)
{
    return 1/std::sqrt(2*pi*sigma)*std::exp(-0.5*(x - u)*(x - u)/sigma);
}
