#include "cnn.h"

RL::Conv2D::Conv2D(std::size_t inputSize_, std::size_t filterSize_,
               std::size_t paddingSize_, std::size_t stride_, std::size_t channel_,
               std::size_t filterNum_)
    :inputSize(inputSize_), filterSize(filterSize_),
      filterNum(filterNum_), paddingSize(paddingSize_),
      stride(stride_), channel(channel_),
      filters(filterNum_, Filter(filterSize_)),
      dFilters(filterNum_, Filter(filterSize_)),
      sFilters(filterNum_, Filter(filterSize_))

{
    outputSize = (inputSize + filterSize - 2 * paddingSize) / stride + 1;
    out = std::vector<Mat>(filterNum_*channel, Mat(outputSize, Vec(outputSize, 0)));
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (std::size_t k = 0; k < filters.size(); k++) {
        filters[k].random();
    }
}

void RL::Conv2D::forward(const std::vector<RL::Mat> &x)
{
    /* (filters, channels, height, width) */
    std::size_t size_ = inputSize + paddingSize - filterSize;
    /* filters */
    for (std::size_t h = 0; h < filterNum; h++) {
        /* channels */
        for (std::size_t k = 0; k < channel; k++) {
            /* height */
            for (std::size_t i = 0; i < size_; i+=stride) {
                /* width */
                for (std::size_t j = 0; j < size_; j+=stride) {
                    std::size_t oi = i / filterSize;
                    std::size_t oj = j / filterSize;
                    /* paddings offset */
                    std::size_t fi0 = i < paddingSize ? paddingSize - i : 0;
                    std::size_t fj0 = j < paddingSize ? paddingSize - j : 0;
                    std::size_t fiEnd = (i + filterSize > inputSize) ? inputSize - i : filterSize;
                    std::size_t fjEnd = (j + filterSize > inputSize) ? inputSize - j : filterSize;
                    /* convolution */
                    for (std::size_t fi = fi0; fi < fiEnd; fi++) {
                        for (std::size_t fj = fj0; fj < fjEnd; fj++) {
                            /* sliding window */
                            std::size_t xi = i < paddingSize ? i : i + fi;
                            std::size_t xj = j < paddingSize ? j : j + fj;
                            out[h*filterNum + k][oi][oj] += filters[h].w[fi][fj] * x[k][xi][xj];
                        }
                    }
                    /* activate */
                    out[h*filterNum + k][oi][oj] = Relu::_(out[h*filterNum + k][oi][oj] + filters[h].b[0]);
                }
            }
        }
    }
    return;
}

void RL::Conv2D::backward()
{

    return;
}

void RL::Conv2D::RMSProp(double rho, double learningRate)
{
    for (std::size_t h = 0; h < filters.size(); h++) {
        Optimizer::RMSProp(filters[h].w, sFilters[h].w, dFilters[h].w, rho, learningRate);
        Optimizer::RMSProp(filters[h].b, sFilters[h].b, dFilters[h].b, rho, learningRate);
        dFilters[h].zero();
    }
    return;
}

RL::MaxPooling::MaxPooling(std::size_t poolingSize_,
                           std::size_t filterNum_,
                           std::size_t channel_,
                           std::size_t inputSize_)
    :inputSize(inputSize_),filterNum(filterNum_),
      channel(channel_),poolingSize(poolingSize_)
{
    std::size_t outputSize = inputSize_ / poolingSize;
    out = std::vector<Mat>(filterNum_*channel_, Mat(outputSize, Vec(outputSize, 0)));
}

void RL::MaxPooling::forward(const std::vector<RL::Mat> &x)
{
    /* filters */
    for (std::size_t h = 0; h < filterNum; h++) {
        /* channels */
        for (std::size_t k = 0; k < channel; k++) {
            /* height */
            for (std::size_t i = 0; i < inputSize; i+= poolingSize) {
                /* width */
                for (std::size_t j = 0; j < inputSize; j+= poolingSize) {
                    /* sliding window */
                    std::size_t oi = i % poolingSize;
                    std::size_t oj = j % poolingSize;
                    double maxValue = 0;
                    for (std::size_t pi = 0; pi < poolingSize; pi++) {
                        for (std::size_t pj = 0; pj < poolingSize; pj++) {
                            std::size_t xi = i + pi;
                            std::size_t xj = j + pj;
                            if (x[h*filterNum + k][xi][xj] > maxValue) {
                                maxValue = x[h*filterNum + k][xi][xj];
                            }
                        }
                    }
                    out[h*filterNum + k][oi][oj] = maxValue;
                }
            }
        }
    }
    return;
}

RL::AveragePooling::AveragePooling(std::size_t poolingSize_,
                                   std::size_t filterNum_,
                                   std::size_t channel_,
                                   std::size_t inputSize_)
    :inputSize(inputSize_),filterNum(filterNum_),
      channel(channel_),poolingSize(poolingSize_)
{
    std::size_t outputSize = inputSize_ / poolingSize;
    out = std::vector<Mat>(filterNum_*channel_, Mat(outputSize, Vec(outputSize, 0)));
}
void RL::AveragePooling::forward(const std::vector<RL::Mat> &x)
{
    /* filters */
    for (std::size_t h = 0; h < filterNum; h++) {
        /* channels */
        for (std::size_t k = 0; k < channel; k++) {
            /* height */
            for (std::size_t i = 0; i < inputSize; i+= poolingSize) {
                /* width */
                for (std::size_t j = 0; j < inputSize; j+= poolingSize) {
                    /* sliding window */
                    std::size_t oi = i % poolingSize;
                    std::size_t oj = j % poolingSize;
                    double s = 0;
                    for (std::size_t pi = 0; pi < poolingSize; pi++) {
                        for (std::size_t pj = 0; pj < poolingSize; pj++) {
                            std::size_t xi = i + pi;
                            std::size_t xj = j + pj;
                            s += x[h*filterNum + k][xi][xj];
                        }
                    }
                    out[h*filterNum + k][oi][oj] = s / (poolingSize * poolingSize);
                }
            }
        }
    }
    return;
}
