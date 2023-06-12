#ifndef CONV2D_HPP
#define CONV2D_HPP
#include "tensor.hpp"
#include "activate.h"


inline void conv2d(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
{
    /* output */
    for (int n = 0; n < y.shape[0]; n++) {
        for (int i = 0; i < y.shape[1]; i++) {
            for (int j = 0; j < y.shape[2]; j++) {
                /* kernels */
                for (int c = 0; c < kernels.shape[1]; c++) {
                    for (int h = 0; h < kernels.shape[2]; h++) {
                        for (int k = 0; k < kernels.shape[3]; k++) {
                            /* map to input  */
                            int row = h + i*stride - padding;
                            int col = k + j*stride - padding;
                            if (row < 0 || row >= x.shape[1] ||
                                    col < 0 || col >= x.shape[2]) {
                                continue;
                            }
                            /* sum up all convolution result */
                            y(n, i, j) += kernels(n, c, h, k)*x(c, row, col);
                        }
                    }
                }
            }
        }
    }
    return;
}
class Conv2dParam
{
public:
    /* conv */
    int inChannels;
    int outChannels;
    int kernelSize;
    int stride;
    int padding;
    bool bias;
    /* i/o */
    int hi;
    int wi;
    int ho;
    int wo;
    /* grad */
    bool withgrad;
public:
    Conv2dParam()
        :inChannels(0),outChannels(0),kernelSize(0),stride(0),padding(0),
         hi(0),wi(0),ho(0),wo(0),withgrad(false){}
    explicit Conv2dParam(const Conv2dParam &param)
        : inChannels(param.inChannels),outChannels(param.outChannels),kernelSize(param.kernelSize),
          stride(param.stride),padding(param.padding),bias(param.bias),
          hi(param.hi),wi(param.wi),ho(param.ho),wo(param.wo),withgrad(param.withgrad){}
    explicit Conv2dParam(int inChannels_,
                         int h,
                         int w,
                         int outChannels_ ,
                         int kernelSize_=3,
                         int stride_=1,
                         int padding_=0,
                         bool bias_=false,
                         bool withgrad_=false):
        inChannels(inChannels_),outChannels(outChannels_),kernelSize(kernelSize_),
    stride(stride_),padding(padding_),bias(bias_),
    hi(h),wi(w),withgrad(withgrad_){}
};

template <typename FnActive>
class Conv2d : public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    class Grad : public Conv2dParam
    {
    public:
        /* grad */
        Tensor dkernels;
        Tensor db;
        Tensor delta;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            dkernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
            if (bias == true) {
                db = Tensor(outChannels, kernelSize, kernelSize);
            }
            delta = Tensor(outChannels, ho, wo);
        }

        inline Tensor& loss() {return delta;}

        void backward(const Conv2d &layer, Tensor &delta_)
        {
            for (int n = 0; n < delta_.shape[0]; n++) {
                for (int i = 0; i < delta_.shape[1]; i++) {
                    for (int j = 0; j < delta_.shape[2]; j++) {

                        int h0 = (i - kernelSize + 1)/stride;
                        h0 = h0 > 0 ? std::ceil(h0):0;
                        int hn = i/stride;
                        hn = hn < ho ? std::floor(hn):ho;

                        int k0 = (j - kernelSize + 1)/stride;
                        k0 = k0 > 0 ? std::ceil(k0):0;
                        int kn = j/stride;
                        kn = kn < wo ? std::floor(kn):wo;

                        for (int c = 0; c < layer.kernels.shape[0]; c++) {
                            for (int h = h0; h < hn; h++) {
                                for (int k = k0; k < kn; k++) {
                                    delta_(n, i, j) += layer.kernels(n, c, i - h*stride, j - k*stride)*delta(c, h, k);
                                }
                            }
                        }
                    }
                }
            }
            return;
        }

        void eval(const Tensor &x, Tensor &o)
        {
            Tensor &dy = o;
            FnActive::df(dy);
            dy *= delta;
            /* db */
            if (bias == true) {
                db += dy;
            }
            /* dkernel */
            for (int n = 0; n < x.shape[0]; n++) {
                for (int i = 0; i < x.shape[1]; i++) {
                    for (int j = 0; j < x.shape[2]; j++) {

                        int h0 = (i - kernelSize + 1)/stride;
                        h0 = h0 > 0 ? std::ceil(h0):0;
                        int hn = i/stride;
                        hn = hn < ho ? std::floor(hn):ho;

                        int k0 = (j - kernelSize + 1)/stride;
                        k0 = k0 > 0 ? std::ceil(k0):0;
                        int kn = j/stride;
                        kn = kn < wo ? std::floor(kn):wo;

                        for (int c = 0; c < dkernels.shape[1]; c++) {
                            for (int h = h0; h < hn; h++) {
                                for (int k = k0; k < kn; k++) {
                                    dkernels(n, c, h, k) += x(n, i, j)*dy(n, h, k);
                                }
                            }
                        }
                    }
                }
            }
            return;
        }

    };

    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        Optimizer optKernels;
        Optimizer optB;
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const Conv2d &layer)
        {
            optKernels = Optimizer(layer.kernels.shape);
            if (layer.bias == true) {
                optB = Optimizer(layer.b.shape);
            }
        }
        void operator()(Conv2d& layer, Grad& grad, float learningRate)
        {
            optKernels(layer.kernels, grad.dkernels, learningRate);
            if (layer.bias == true) {
                optB(layer.b, grad.db, learningRate);
            }
            return;
        }
    };
public:
    /* (N, c, kernelSize, kernelSize) */
    Tensor kernels;
    /* (N, ho, wo) */
    Tensor o;
    /* (N, ho, wo) */
    Tensor b;
public:
    Conv2d(){}
    explicit Conv2d(int inChannels_,
                    int h,
                    int w,
                    int outChannels_,
                    int kernelSize_=3,
                    int stride_=1,
                    int padding_=0,
                    bool bias_=false,
                    bool withgrad_=false):
        Conv2dParam(inChannels_, h, w, outChannels_, kernelSize_, stride_, padding_, bias_, withgrad_)
    {
        kernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
        uniformRand(kernels, -1, 1);
        ho = std::floor((hi - kernelSize + 2*padding)/stride) + 1;
        wo = std::floor((wi - kernelSize + 2*padding)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        if (bias == true) {
            b = Tensor(outChannels, kernelSize, kernelSize);
            uniformRand(b, -1, 1);
        }
    }

    Tensor& forward(const Tensor &x)
    {
        /* conv */
        o.zero();
        conv2d(o, kernels, x, stride, padding);
        /* bias */
        if (bias == true) {
            o += b;
        }
        /* activate */
        FnActive::f(o);
        return o;
    }
};

class MaxPooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    /* grad */
    class Grad : public Conv2dParam
    {
    public:
        Tensor delta;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            delta = Tensor(outChannels, ho, wo);
        }
        inline Tensor& loss() {return delta;}
        void backward(MaxPooling2d &layer, Tensor &delta_)
        {
            for (int n = 0; n < delta_.shape[0]; n++) {
                for (int i = 0; i < delta_.shape[1]; i++) {
                    for (int j = 0; j < delta_.shape[2]; j++) {

                        int h0 = (i - kernelSize + 1)/stride;
                        h0 = h0 > 0 ? std::ceil(h0):0;
                        int hn = i/stride;
                        hn = hn < ho ? std::floor(hn):ho;

                        int k0 = (j - kernelSize + 1)/stride;
                        k0 = k0 > 0 ? std::ceil(k0):0;
                        int kn = j/stride;
                        kn = kn < wo ? std::floor(kn):wo;

                        for (int h = h0; h < hn; h++) {
                            for (int k = k0; k < kn; k++) {
                                delta_(n, i, j) += layer.mask(n, h, k)*delta(n, h, k);
                            }
                        }
                    }
                }
            }
            layer.mask.zero();
            return;
        }
        /* no gradient */
        void eval(const Tensor &, const Tensor &){}
    };
    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const MaxPooling2d &){}
        void operator()(MaxPooling2d&, Grad&, float){}
    };
public:
    Tensor o;
    Tensor mask;
public:
    MaxPooling2d(){}
    explicit MaxPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        Conv2dParam(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        mask = Tensor(outChannels, ho, wo);
    }

    Tensor& forward(const Tensor &x)
    {
        /* input shape is same as output shape */
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    float maxValue = 0;
                    for (int h = 0; h < kernelSize; h++) {
                        for (int k = 0; k < kernelSize; k++) {
                            float value = x(n, h + i*stride, k + j*stride);
                            if (value > maxValue) {
                                maxValue = value;
                                mask(n, h, k) = 1;
                            }
                        }
                    }
                    o(n, i, j) = maxValue;
                }
            }
        }
        return o;
    }

};

class AvgPooling2d: public Conv2dParam
{
public:
    using ParamType = Conv2dParam;
    /* grad */
    class Grad : public Conv2dParam
    {
    public:
        Tensor delta;
    public:
        Grad(){}
        explicit Grad(const Conv2dParam &param)
            :Conv2dParam(param)
        {
            delta = Tensor(outChannels, ho, wo);
        }
        inline Tensor& loss() {return delta;}
        void backward(AvgPooling2d &layer, Tensor &delta_)
        {
            /* delta_: previous delta, the shape is same as delta and output */
            for (int n = 0; n < delta_.shape[0]; n++) {
                for (int i = 0; i < delta_.shape[1]; i++) {
                    for (int j = 0; j < delta_.shape[2]; j++) {

                        int h0 = (i - kernelSize + 1)/stride;
                        h0 = h0 > 0 ? std::ceil(h0):0;
                        int hn = i/stride;
                        hn = hn < ho ? std::floor(hn):ho;

                        int k0 = (j - kernelSize + 1)/stride;
                        k0 = k0 > 0 ? std::ceil(k0):0;
                        int kn = j/stride;
                        kn = kn < wo ? std::floor(kn):wo;

                        for (int h = h0; h < hn; h++) {
                            for (int k = k0; k < kn; k++) {
                                delta_(n, i, j) += delta(n, h, k);
                            }
                        }
                    }
                }
            }
            return;
        }
        /* no gradient */
        void eval(const Tensor &, const Tensor &){}
    };

    /* optimizer */
    template<typename Optimizer>
    class OptimizeBlock
    {
    public:
        OptimizeBlock(){}
        explicit OptimizeBlock(const AvgPooling2d &){}
        void operator()(AvgPooling2d&, Grad&, float){}
    };
public:
    Tensor o;
public:
    AvgPooling2d(){}
    explicit AvgPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        Conv2dParam(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
    }

    Tensor& forward(const Tensor &x)
    {
        /* conv */
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    float u = 0;
                    for (int h = 0; h < kernelSize; h++) {
                        for (int k = 0; k < kernelSize; k++) {
                            u += x(n, h + i*stride, k + j*stride);
                        }
                    }
                    o(n, i, j) = u/(kernelSize*kernelSize);
                }
            }
        }
        return o;
    }

};

#endif // CONV2D_HPP
