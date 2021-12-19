#include "util.h"

std::default_random_engine RL::Rand::engine(std::random_device{}());

int RL::argmax(const Vec &x)
{
    int index = 0;
    double maxValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (maxValue < x[i]) {
            maxValue = x[i];
            index = i;
        }
    }
    return index;
}

int RL::argmin(const Vec &x)
{
    int index = 0;
    double minValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (minValue > x[i]) {
            minValue = x[i];
            index = i;
        }
    }
    return index;
}

double RL::max(const Vec &x)
{
    double maxValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (maxValue < x[i]) {
            maxValue = x[i];
        }
    }
    return maxValue;
}

double RL::min(const Vec &x)
{
    double minValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (minValue > x[i]) {
            minValue = x[i];
        }
    }
    return minValue;
}

void RL::zscore(Vec &x)
{
    /* sigma */
    double sigma = variance(x);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] /= sigma;
    }
    return;
}

double RL::mean(const Vec &x)
{
    return sum(x) / double(x.size());
}

double RL::sum(const Vec &x)
{
    double s = 0;
    for (auto& value : x) {
        s += value;
    }
    return s;
}

void RL::normalize(Vec &x)
{
    double minValue = x[0];
    double maxValue = x[0];
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

double RL::variance(const Vec &x)
{
    /* expectation */
    double u = mean(x);
    double sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sqrt(sigma / x.size());
}

double RL::variance(const Vec &x, double u)
{
    double sigma = 0;
    /* sigma */
    for (std::size_t i = 0 ; i < x.size(); i++) {
        sigma += (x[i] - u) * (x[i] - u);
    }
    return sqrt(sigma / x.size());
}

double RL::dot(const Vec& x1, const Vec& x2)
{
    double s = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
        s += x1[i] * x2[i];
    }
    return s;
}

double RL::covariance(const Vec &x1, const Vec &x2)
{
    double u = mean(x1);
    double v = mean(x2);
    double covar = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
         covar += (x1[i] - u) * (x2[i] - v);
    }
    return covar;
}

double RL::clip(double x, double sup, double inf)
{
    double y = x;
    if (x < sup) {
        y = sup;
    } else if (x > inf) {
        y = inf;
    }
    return y;
}
