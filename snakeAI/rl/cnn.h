#ifndef CNN_H
#define CNN_H
#include "util.h"
#include "activate.h"
#include "optimizer.h"
#include "loss.h"
#include <cstring>
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
    std::size_t size_;
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
    std::vector<Mat> filters;
    std::vector<Mat> dFilters;
    std::vector<Mat> sFilters;
    /*
        (filters, channels, height, width)
        output size = (W - F + 2P) / s
     */
    std::vector<std::vector<Mat> > out;
public:
    static void toMat(Image &img, std::vector<Mat> &dst);
    Conv(){}
    Conv(std::size_t inputSize_ = 32,
         std::size_t filterSize_ = 3,
         std::size_t paddingSize_ = 2,
         std::size_t stride_ = 2,
         std::size_t channel_ = 3,
         std::size_t filterNum = 12);
    void forward(const std::vector<Mat> &x);
    void backward();
    void RMSProp(double rho, double learningRate);
};

class MaxPooling
{
public:
    std::vector<Mat> out;
public:
    void forward(const std::vector<Mat> &x);
    void backward();
    void gradient();
};

class AveragePooling
{
public:
    std::vector<Mat> out;
public:
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
