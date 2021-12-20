#include "cnn.h"

void RL::Conv::toMat(Image &img, std::vector<RL::Mat> &dst)
{
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            unsigned char *rgb = img.at(i, j);
            dst[0][i][j] = double(rgb[0])/255.0;
            dst[1][i][j] = double(rgb[1])/255.0;
            dst[2][i][j] = double(rgb[2])/255.0;
        }
    }
    return;
}

RL::Conv::Conv(std::size_t inputSize_, std::size_t filterSize_,
               std::size_t paddingSize_, std::size_t stride_, std::size_t channel_,
               std::size_t filterNum_)
    :inputSize(inputSize_),filterSize(filterSize_),paddingSize(paddingSize_),stride(stride_),channel(channel_),
      filters(filterNum_, Filter(filterSize_)),
      dFilters(filterNum_, Filter(filterSize_)),
      sFilters(filterNum_, Filter(filterSize_))

{
    outputSize = (inputSize + filterSize - 2 * paddingSize) / stride;
    out = std::vector<std::vector<Mat> >(filterNum_, std::vector<Mat>(channel_, Mat(outputSize, Vec(outputSize, 0))));
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (std::size_t k = 0; k < filters.size(); k++) {
        filters[k].random();
    }
}

void RL::Conv::forward(const std::vector<RL::Mat> &x)
{
    /* (filters, channels, height, width) */
    /* filters */
    for (std::size_t h = 0; h < out.size(); h++) {
        /* channels */
        for (std::size_t k = 0; k < out[0].size(); k++) {
            /* height */
            for (std::size_t i = 0; i + filterSize - paddingSize < x[k].size(); i+=stride) {
                /* width */
                for (std::size_t j = 0; j + filterSize - paddingSize < x[k][0].size(); j+=stride) {
                    /* sliding window */
                    std::size_t oi = i % filterSize;
                    std::size_t oj = j % filterSize;
                    /* paddings offset */
                    std::size_t fi0 = i == 0 ? paddingSize : 0;
                    std::size_t fj0 = j == 0 ? paddingSize : 0;
                    std::size_t fiEnd = (i + filterSize - paddingSize) == x[0].size() ? filterSize - paddingSize : filterSize;
                    std::size_t fjEnd = (i + filterSize - paddingSize) == x[0][0].size() ? filterSize - paddingSize : filterSize;
                    for (std::size_t fi = fi0; fi < fiEnd; fi++) {
                        for (std::size_t fj = fj0; fj < fjEnd; fj++) {
                            std::size_t xi = i + fi;
                            std::size_t xj = j + fj;
                            out[h][k][oi][oj] += filters[h].w[fi][fj] * x[k][xi][xj];
                        }
                    }
                    out[h][k][oi][oj] = Relu::_(out[h][k][oi][oj] + filters[h].b[0]);
                }
            }
        }
    }
    return;
}

void RL::Conv::backward()
{

    return;
}

void RL::Conv::RMSProp(double rho, double learningRate)
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
    :poolingSize(poolingSize_)
{
    std::size_t outputSize = inputSize_ / poolingSize;
    out = std::vector<std::vector<Mat> >(filterNum_, std::vector<Mat>(channel_, Mat(outputSize, Vec(outputSize, 0))));
}

void RL::MaxPooling::forward(const std::vector<std::vector<RL::Mat> > &x)
{
    /* filters */
    for (std::size_t h = 0; h < x.size(); h++) {
        /* channels */
        for (std::size_t k = 0; k < x[0].size(); k++) {
            /* height */
            for (std::size_t i = 0; i + poolingSize < x[0][0].size(); i+= poolingSize) {
                /* width */
                for (std::size_t j = 0; j + poolingSize< x[0][0][0].size(); j+= poolingSize) {
                    /* sliding window */
                    std::size_t oi = i % poolingSize;
                    std::size_t oj = j % poolingSize;
                    double maxValue = 0;
                    for (std::size_t fi = 0; fi < poolingSize; fi++) {
                        for (std::size_t fj = 0; fj < poolingSize; fj++) {
                            std::size_t xi = i + fi;
                            std::size_t xj = j + fj;
                            if (x[h][k][xi][xj] > maxValue) {
                                maxValue = x[h][k][xi][xj];
                            }
                        }
                    }
                    out[h][k][oi][oj] = maxValue;
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
    :poolingSize(poolingSize_)
{
    std::size_t outputSize = inputSize_ / poolingSize;
    out = std::vector<std::vector<Mat> >(filterNum_, std::vector<Mat>(channel_, Mat(outputSize, Vec(outputSize, 0))));
}
void RL::AveragePooling::forward(const std::vector<std::vector<RL::Mat> > &x)
{
    /* filters */
    for (std::size_t h = 0; h < x.size(); h++) {
        /* channels */
        for (std::size_t k = 0; k < x[0].size(); k++) {
            /* height */
            for (std::size_t i = 0; i + poolingSize < x[0][0].size(); i+= poolingSize) {
                /* width */
                for (std::size_t j = 0; j + poolingSize< x[0][0][0].size(); j+= poolingSize) {
                    /* sliding window */
                    std::size_t oi = i % poolingSize;
                    std::size_t oj = j % poolingSize;
                    double s = 0;
                    for (std::size_t fi = 0; fi < poolingSize; fi++) {
                        for (std::size_t fj = 0; fj < poolingSize; fj++) {
                            std::size_t xi = i + fi;
                            std::size_t xj = j + fj;
                            s += x[h][k][xi][xj];
                        }
                    }
                    out[h][k][oi][oj] = s / (poolingSize * poolingSize);
                }
            }
        }
    }
    return;
}
