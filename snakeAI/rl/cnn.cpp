#include "cnn.h"

RL::Conv2D::Conv2D(int inputSize_, int filterSize_, int paddingSize_,
                   int stride_, int channel_, int filterNum_)
    :inputSize(inputSize_), filterSize(filterSize_),
      filterNum(filterNum_), paddingSize(paddingSize_),
      stride(stride_), channel(channel_),
      filters(filterNum_, Filter(filterSize_)),
      dFilters(filterNum_, Filter(filterSize_)),
      sFilters(filterNum_, Filter(filterSize_))

{
    outputSize = (inputSize - filterSize + 2*paddingSize)/stride + 1;
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (int k = 0; k < filters.size(); k++) {
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
    for (int i = 0; i < filterNum; i++) {
        /* channels */
        for (int j = 0; j < channel; j++) {
            conv(out[i], filters[i].w, x[j]);
        }
        /* activate */
        for (int row = 0; row < outputSize; row++) {
            for (int col = 0; col < outputSize; col++) {
                out[i][row][col] = Relu::_(out[i][row][col] + filters[i].b[0]);
            }
        }
    }
    return;
}

void RL::Conv2D::convNoPad(RL::Mat &y, const RL::Mat &kernel, const RL::Mat &x)
{
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            for (int h = 0; h < filterSize; h++) {
                for (int k = 0; k < filterSize; k++) {
                    y[i][j] += kernel[h][k]*x[i + h][j + k];
                }
            }
        }
    }
    return;
}

void RL::Conv2D::conv(RL::Mat &y, const RL::Mat &kernel, const RL::Mat &x)
{
    /* height */
    for (int i = 0; i < inputSize; i+=stride) {
        /* width */
        for (int j = 0; j < inputSize; j+=stride) {
            conv_(i, j, y, kernel, x);
        }
    }
    return;
}

void RL::Conv2D::conv_(int ik, int jk, RL::Mat &y, const RL::Mat &kernel, const RL::Mat &x)
{
    /*
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    */
    int oi = ik/stride;
    int oj = jk/stride;
    if (oi >= outputSize) {
        return;
    }
    if (oj >= outputSize) {
        return;
    }
    if (ik + filterSize > inputSize + paddingSize) {
        return;
    }
    if (jk + filterSize > inputSize + paddingSize) {
        return;
    }
    /* paddings offset */
    int i0 = ik < paddingSize ? paddingSize - ik : 0;
    int j0 = jk < paddingSize ? paddingSize - jk : 0;
    int ie = (ik + filterSize > inputSize) ? inputSize - ik : filterSize;
    int je = (jk + filterSize > inputSize) ? inputSize - jk : filterSize;
    /* convolution */
    for (int i = i0; i < ie; i++) {
        for (int j = j0; j < je; j++) {
            int row = ik < paddingSize ? i : ik + i;
            int col = jk < paddingSize ? j : jk + j;
            y[oi][oj] += kernel[i][j]*x[row][col];
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
    for (int h = 0; h < filters.size(); h++) {
        Optimizer::RMSProp(filters[h].w, sFilters[h].w, dFilters[h].w, rho, learningRate);
        Optimizer::RMSProp(filters[h].b, sFilters[h].b, dFilters[h].b, rho, learningRate);
        dFilters[h].zero();
    }
    return;
}

void RL::Conv2D::test()
{
    /* input */
    Mat x(6, Vec(6, 1));
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            if (i == 0 || j == 0 || i == 5 || j == 5) {
                x[i][j] = 2;
            }
        }
    }
    Conv2D layer(6, 3, 1, 1, 1, 1);
    /* kernel */
    layer.filters[0].w = Mat(3, Vec(3, 1));
    /* convolution */
    layer.conv(layer.out[0], layer.filters[0].w, x);
    /* show */
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            std::cout<<x[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    for (int i = 0; i < layer.out[0].size(); i++) {
        for (int j = 0; j < layer.out[0][0].size(); j++) {
            std::cout<<layer.out[0][i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
}

RL::MaxPooling::MaxPooling(int poolingSize_,
                           int filterNum_,
                           int channel_,
                           int inputSize_)
    :inputSize(inputSize_),filterNum(filterNum_),
      channel(channel_),poolingSize(poolingSize_)
{
    int outputSize = inputSize_ / poolingSize;
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
}

void RL::MaxPooling::forward(const std::vector<RL::Mat> &x)
{
    /*
       input: (filters, height, width)
       output: (filters, height, width)
    */
    /* filters */
    for (int h = 0; h < filterNum; h++) {
        /* height */
        for (int i = 0; i < inputSize; i+= poolingSize) {
            /* width */
            for (int j = 0; j < inputSize; j+= poolingSize) {
                /* sliding window */
                int oi = i / poolingSize;
                int oj = j / poolingSize;
                double maxValue = 0;
                for (int pi = 0; pi < poolingSize; pi++) {
                    for (int pj = 0; pj < poolingSize; pj++) {
                        int xi = i + pi;
                        int xj = j + pj;
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

RL::AveragePooling::AveragePooling(int poolingSize_,
                                   int filterNum_,
                                   int channel_,
                                   int inputSize_)
    :inputSize(inputSize_),filterNum(filterNum_),
      channel(channel_),poolingSize(poolingSize_)
{
    int outputSize = inputSize_ / poolingSize;
    out = std::vector<Mat>(filterNum_, Mat(outputSize, Vec(outputSize, 0)));
}
void RL::AveragePooling::forward(const std::vector<RL::Mat> &x)
{
    /* filters */
    for (int h = 0; h < filterNum; h++) {
        /* height */
        for (int i = 0; i < inputSize; i+= poolingSize) {
            /* width */
            for (int j = 0; j < inputSize; j+= poolingSize) {
                /* sliding window */
                int oi = i / poolingSize;
                int oj = j / poolingSize;
                double s = 0;
                for (int pi = 0; pi < poolingSize; pi++) {
                    for (int pj = 0; pj < poolingSize; pj++) {
                        int xi = i + pi;
                        int xj = j + pj;
                        s += x[h][xi][xj];
                    }
                }
                out[h][oi][oj] = s / (poolingSize * poolingSize);
            }
        }
    }
    return;
}
