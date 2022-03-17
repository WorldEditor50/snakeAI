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
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (std::size_t k = 0; k < filters.size(); k++) {
        filters[k].random();
    }
}

void RL::Conv2D::forward(const std::vector<RL::Mat> &x)
{
    /*
       input: (channels, height, width)
       output: (filters, height, width)
       output_i = relu(input_r*kernel_i + input_g*kernel_i + input_b*kernel_i + b_i)
    */

    /* filters */
    for (std::size_t i = 0; i < filterNum; i++) {
        /* channels */
        for (std::size_t j = 0; j < channel; j++) {
            conv(x[j], filters[i].w, out[i]);
        }
        /* activate */
        for (std::size_t row = 0; row < out[0].size(); row++) {
            for (std::size_t col = 0; col < out[0][0].size(); col++) {
                out[i][row][col] = Relu::_(out[i][row][col] + filters[i].b[0]);
            }
        }
    }
    return;
}

void RL::Conv2D::conv(const RL::Mat &x, const RL::Mat &kernel, RL::Mat &y)
{
    /* height */
    for (std::size_t i = 0; i < x.size(); i+=stride) {
        /* width */
        for (std::size_t j = 0; j < x[0].size(); j+=stride) {
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
                    y[oi][oj] += kernel[fi][fj] * x[xi][xj];
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
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
}

void RL::MaxPooling::forward(const std::vector<RL::Mat> &x)
{
    /*
       input: (filters, height, width)
       output: (filters, height, width)
    */
    /* filters */
    for (std::size_t h = 0; h < filterNum; h++) {
        /* height */
        for (std::size_t i = 0; i < inputSize; i+= poolingSize) {
            /* width */
            for (std::size_t j = 0; j < inputSize; j+= poolingSize) {
                /* sliding window */
                std::size_t oi = i / poolingSize;
                std::size_t oj = j / poolingSize;
                double maxValue = 0;
                for (std::size_t pi = 0; pi < poolingSize; pi++) {
                    for (std::size_t pj = 0; pj < poolingSize; pj++) {
                        std::size_t xi = i + pi;
                        std::size_t xj = j + pj;
                        if (x[h][xi][xj] > maxValue) {
                            maxValue = x[h][xi][xj];
                        }
                    }
                }
                out[h][oi][oj] = maxValue;
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
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
}
void RL::AveragePooling::forward(const std::vector<RL::Mat> &x)
{
    /* filters */
    for (std::size_t h = 0; h < filterNum; h++) {
        /* height */
        for (std::size_t i = 0; i < inputSize; i+= poolingSize) {
            /* width */
            for (std::size_t j = 0; j < inputSize; j+= poolingSize) {
                /* sliding window */
                std::size_t oi = i / poolingSize;
                std::size_t oj = j / poolingSize;
                double s = 0;
                for (std::size_t pi = 0; pi < poolingSize; pi++) {
                    for (std::size_t pj = 0; pj < poolingSize; pj++) {
                        std::size_t xi = i + pi;
                        std::size_t xj = j + pj;
                        s += x[h][xi][xj];
                    }
                }
                out[h][oi][oj] = s / (poolingSize * poolingSize);
            }
        }
    }
    return;
}
