#ifndef CNN_H
#define CNN_H
#include "util.h"
#include "activate.h"
#include "optimizer.h"
#include "loss.h"
#include <cstring>
#include <iostream>

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

class Image
{
public:
    int width;
    int height;
    int channel;
    unsigned char *ptr;
    int size_;
public:
    Image():width(0),height(0),channel(3),ptr(nullptr),size_(0){}
    Image(int w, int h, int c):width(w),height(h),channel(c)
    {
        int widthstep = (width * channel + 3)/4*4;
        size_ = widthstep * height;
        ptr = new unsigned char[size_];
    }
    Image(const Image &img)
        :width(img.width),height(img.height),channel(img.channel),size_(img.size_)
    {
        ptr = new unsigned char[size_];
        memcpy(ptr, img.ptr, size_);
    }
    Image &operator=(const Image &img)
    {
        if (this == &img) {
            return *this;
        }
        width = img.width;
        height = img.height;
        channel = img.channel;
        size_ = img.size_;
        ptr = new unsigned char[size_];
        memcpy(ptr, img.ptr, size_);
        return *this;
    }
    Image(Image &&img)
        :width(img.width),height(img.height),channel(img.channel),
          ptr(img.ptr),size_(img.size_)
    {
        img.width = 0;
        img.height = 0;
        img.channel = 0;
        img.ptr = nullptr;
        img.size_ = 0;
    }
    Image &operator=(Image &&img)
    {
        if (this == &img) {
            return *this;
        }
        width = img.width;
        height = img.height;
        channel = img.channel;
        size_ =img.size_;
        ptr = img.ptr;
        img.width = 0;
        img.height = 0;
        img.channel = 0;
        img.ptr = nullptr;
        img.size_ = 0;
        return *this;
    }
    ~Image()
    {
        clear();
    }
    void clear()
    {
        if (ptr != nullptr) {
            delete [] ptr;
            ptr = nullptr;
            width = 0;
            height = 0;
            channel = 0;
            size_ = 0;
        }
        return;
    }
    inline unsigned char* scanline(int i)
    {
        return ptr + i*(width*channel + 3)/4*4;
    }
    inline unsigned char* at(int i, int j)
    {
        return ptr + i*(width*channel + 3)/4*4 + j*channel;
    }
    static void toMat(Image &img, std::vector<RL::Mat> &dst)
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
};
class Filter
{
public:
    Mat w;
    Vec b;
public:
    Filter(){}
    Filter(int size_)
    {
        w = Mat(size_, Vec(size_, 0));
        b = Vec(1, 0);
    }
    void random()
    {
        std::uniform_real_distribution<double> uniform(-1, 1);
        for (int i = 0; i < w.size(); i++) {
            for (int j = 0; j < w[0].size(); j++) {
                w[i][j] = uniform(Rand::engine);
            }
        }
        b[0] = uniform(Rand::engine);
        return;
    }
    void zero()
    {
        std::uniform_real_distribution<double> uniform(-1, 1);
        for (int i = 0; i < w.size(); i++) {
            for (int j = 0; j < w[0].size(); j++) {
                w[i][j] = 0;
            }
        }
        b[0] = 0;
        return;
    }
};

class Conv2D
{
public:
    int inputSize;
    int filterSize;
    int filterNum;
    int paddingSize;
    int outputSize;
    int stride;
    int channel;
    std::vector<Filter> filters;
    std::vector<Filter> dFilters;
    std::vector<Filter> sFilters;
    /*
        (filters, height, width)
        output size = (W - F + 2P) / s
     */
    std::vector<Mat> out;
public:
    Conv2D(){}
    explicit Conv2D(int inputSize_ = 32,
         int filterSize_ = 3,
         int paddingSize_ = 2,
         int stride_ = 2,
         int channel_ = 3,
         int filterNum = 12);
    void forward(const std::vector<Mat> &x);
    void convNoPad(Mat &y, const Mat &kernel, const Mat &x);
    void conv(Mat &y, const Mat &kernel, const Mat &x);
    void conv_(int ik, int jk, Mat &y, const Mat &kernel, const Mat &x);
    void backward();
    void RMSProp(double rho, double learningRate);
    static void test();
};

class MaxPooling
{
public:
    int inputSize;
    int filterNum;
    int channel;
    int poolingSize;
    std::vector<RL::Mat> out;
public:
    explicit MaxPooling(int poolingSize_,
                        int filterNum_,
                        int channel_,
                        int inputSize_);
    void forward(const std::vector<Mat> &x);
    void backward();
    void gradient();
};

class AveragePooling
{
public:
    int inputSize;
    int filterNum;
    int channel;
    int poolingSize;
    std::vector<RL::Mat> out;
public:
    explicit AveragePooling(int poolingSize_,
                            int filterNum_,
                            int channel_,
                            int inputSize_);
    void forward(const std::vector<Mat> &x);
    void backward();
    void gradient();
};

class Dropout
{
public:
    std::vector<Mat> out;
public:
    void forward(const std::vector<Mat> &x);
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
    Conv2D conv1;
    MaxPooling maxpool1;
    Conv2D conv2;
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
