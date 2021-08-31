#include "rl_basic.h"
#include <cmath>

int RL::argmax(const std::vector<double> &x)
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

int RL::argmin(const std::vector<double> &x)
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

double RL::max(const std::vector<double> &x)
{
    double maxValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (maxValue < x[i]) {
            maxValue = x[i];
        }
    }
    return maxValue;
}

double RL::min(const std::vector<double> &x)
{
    double minValue = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (minValue > x[i]) {
            minValue = x[i];
        }
    }
    return minValue;
}

void RL::zscore(std::vector<double> &x)
{
    /* sigma */
    double sigma = variance(x);
    for (std::size_t i = 0 ; i < x.size(); i++) {
        x[i] /= sigma;
    }
    return;
}

double RL::mean(const std::vector<double> &x)
{
    return sum(x) / x.size();
}

double RL::sum(const std::vector<double> &x)
{
    double s = 0;
    for (auto& value : x) {
        s += value;
    }
    return s;
}

void RL::normalize(std::vector<double> &x)
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

double RL::variance(const std::vector<double> &x)
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

double RL::dotProduct(const std::vector<double>& x1, const std::vector<double>& x2)
{
    double s = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
        s += x1[i] * x2[i];
    }
    return s;
}

double RL::covariance(const std::vector<double> &x1, const std::vector<double> &x2)
{
    double u = mean(x1);
    double v = mean(x2);
    double covar = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
         covar += (x1[i] - u) * (x2[i] - v);
    }
    return covar;
}
