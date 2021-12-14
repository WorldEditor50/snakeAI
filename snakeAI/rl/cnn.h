#ifndef CNN_H
#define CNN_H
#include <QImage>
#include "rl_basic.h"
namespace RL {

//using RGB = unsigned int;
#define RGB_MASK   0x00ffffff
#define RED_MASK   0x00ff0000
#define GREEN_MASK 0x0000ff00
#define BLUE_MASK  0x000000ff
class RGB
{
public:
    unsigned int data;
public:
    RGB():data(0){}
    RGB(unsigned char r, unsigned char g, unsigned b)
    {
        int ir = (r & 0xff) << 16;
        int ig = (g & 0xff) << 8;
        int ib = b & 0xff;
        data = (ir + ig + ib)&RGB_MASK;
    }
    RGB(unsigned int rgb):data(rgb){}
    inline unsigned char R(){return ((data&RED_MASK) >> 16)&0xff;}
    inline unsigned char G(){return ((data&GREEN_MASK) >> 8)&0xff;}
    inline unsigned char B(){return (data&BLUE_MASK)&0xff;}
};

class Conv
{
public:
    std::size_t inputSize;
    std::size_t filterSize;
    std::size_t paddingSize;
    std::size_t outputSize;
    std::size_t stride;
    std::size_t channel;
    /* output size = (W - F + 2P) / s */
    std::vector<Mat> filters;
    /* (filters, channels, height, width) */
    std::vector<std::vector<Mat> > data;
public:
    static void red(const QImage &img, Mat &dst);
    static void green(const QImage &img, Mat &dst);
    static void blue(const QImage &img, Mat &dst);
    static void gray(const QImage &img, Mat &dst);
    Conv(){}
    Conv(std::size_t inputSize_ = 32,
         std::size_t filterSize_ = 3,
         std::size_t paddingSize_ = 2,
         std::size_t stride_ = 2,
         std::size_t channel_ = 3,
         std::size_t filterNum = 12);
    void forward(const std::vector<Mat> &x);
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
    void forward(const Mat &x);
    void backward();
    void gradient();
};

}
#endif // CNN_H
