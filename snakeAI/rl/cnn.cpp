#include "cnn.h"

void RL::Conv::red(const QImage &img, Mat &dst)
{
    for (int i = 0; i < img.height(); i++) {
        QRgb *rgb = (QRgb*)img.scanLine(i);
        for (int j = 0; j < img.width(); j++) {
            dst[i][j] = qRed(rgb[j]);
        }
    }
    return;
}

void RL::Conv::green(const QImage &img, Mat &dst)
{
    for (int i = 0; i < img.height(); i++) {
        QRgb *rgb = (QRgb*)img.scanLine(i);
        for (int j = 0; j < img.width(); j++) {
            dst[i][j] = qGreen(rgb[j]);
        }
    }
    return;
}

void RL::Conv::blue(const QImage &img, Mat &dst)
{
    for (int i = 0; i < img.height(); i++) {
        QRgb *rgb = (QRgb*)img.scanLine(i);
        for (int j = 0; j < img.width(); j++) {
            dst[i][j] = qBlue(rgb[j]);
        }
    }
    return;
}

void RL::Conv::gray(const QImage &img, Mat &dst)
{
    for (int i = 0; i < img.height(); i++) {
        QRgb *rgb = (QRgb*)img.scanLine(i);
        for (int j = 0; j < img.width(); j++) {
            dst[i][j] = (qRed(rgb[j]) + qGreen(rgb[j]) + qBlue(rgb[j])) / 3;
        }
    }
    return;
}

RL::Conv::Conv(std::size_t inputSize_, std::size_t filterSize_,
               std::size_t paddingSize_, std::size_t stride_, std::size_t channel_,
               std::size_t filterNum_)
    :inputSize(inputSize_),filterSize(filterSize_),paddingSize(paddingSize_),stride(stride_),
      filters(filterNum_, Mat(filterSize_, Vec(filterSize_, 0)))
{
    outputSize = (inputSize + filterSize - 2 * paddingSize) / stride;
    data = std::vector<std::vector<Mat> >(filterNum_, std::vector<Mat>(3, Mat(outputSize, Vec(outputSize, 0))));
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

}
