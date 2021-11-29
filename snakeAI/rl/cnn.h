#ifndef CNN_H
#define CNN_H
#include "rl_basic.h"
namespace RL {


class Conv
{
public:
    std::size_t imageSize;
    std::size_t filterSize;
    std::size_t paddingSize;
    std::size_t stride;
    std::vector<Mat> filters;
    /* output size = (W - F + 2P) / s */
    Mat O;
public:
    Conv(){}
    Conv(std::size_t imageSize_,
         std::size_t filterSize_,
         std::size_t paddingSize_,
         std::size_t stride_);
    void forward(const Mat &x);
    void backward();
    void gradient();
};

class MaxPooling
{
public:
    Mat O;
public:
    void forward(const Mat &x);
    void backward();
    void gradient();
};

class AveragePooling
{
public:
    Mat O;
public:
    void forward(const Mat &x);
    void backward();
    void gradient();
};

class Dropout
{
public:
    Mat O;
public:
    void forward(const Mat &x);
    void backward();
    void gradient();
};

class FC
{
public:
    Mat W;
    Vec B;
    Mat O;
    Vec E;
public:
    void forward(const Mat &x);
    void backward();
    void gradient();
};

class CNN
{
public:
    Conv conv1;
    MaxPooling maxpool1;
    Conv conv2;
    MaxPooling maxpool2;
    FC fc1;
    FC fc2;
public:
    CNN(){}
    Mat O;
public:
    void forward(const Mat &x);
    void backward();
    void gradient();
};

}
#endif // CNN_H
