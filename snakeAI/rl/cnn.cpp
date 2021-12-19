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
      filters(filterNum_, Mat(filterSize_, Vec(filterSize_, 0))),
      dFilters(filterNum_, Mat(filterSize_, Vec(filterSize_, 0))),
      sFilters(filterNum_, Mat(filterSize_, Vec(filterSize_, 0)))

{
    outputSize = (inputSize + filterSize - 2 * paddingSize) / stride;
    out = std::vector<std::vector<Mat> >(filterNum_, std::vector<Mat>(channel_, Mat(outputSize, Vec(outputSize, 0))));
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (std::size_t k = 0; k < filters.size(); k++) {
        for (std::size_t i = 0; i < filters[0].size(); i++) {
            for (std::size_t j = 0; j < filters[0][0].size(); j++) {
                filters[k][i][j] = uniform(Rand::engine);
            }
        }
    }
}

void RL::Conv::forward(const std::vector<RL::Mat> &x)
{
    /* (filters, channels, height, width) */
    for (std::size_t h = 0; h < out.size(); h++) {
        for (std::size_t k = 0; k < out[0].size(); k++) {
            int fi = 0;
            int oi = 0;
            int fj = 0;
            int oj = 0;
            for (std::size_t i = 0; i < x[k].size(); i++) {
                fi = (i + paddingSize)%filterSize;
                for (std::size_t j = 0; j < x[k][0].size(); j++) {
                    fj = (j + paddingSize)%filterSize;
                    out[h][k][oi][oj] += filters[h][fi][fj] * x[k][i][j];
                }
                out[h][k][oi][oj] = Relu::_(out[h][k][oi][oj]);
                oi += (i % filterSize == 0) ? 1 : 0;
                oj += (i % filterSize == 0) ? 1 : 0;
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
        Optimizer::RMSProp(filters[h], sFilters[h], dFilters[h], rho, learningRate);
    }
    return;
}
