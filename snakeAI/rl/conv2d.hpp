#ifndef CONV2D_HPP
#define CONV2D_HPP
#include "tensor.hpp"
#include "activate.h"
#include "util.hpp"
#include "layer.h"

namespace RL {

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

class iConv2d : public iLayer
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
public:
    iConv2d(){}
    explicit iConv2d(int inChannels_,
                     int h,
                     int w,
                     int outChannels_,
                     int kernelSize_=3,
                     int stride_=1,
                     int padding_=0,
                     bool bias_=false)
        :inChannels(inChannels_), hi(h), wi(w),outChannels(outChannels_),
    kernelSize(kernelSize_), stride(stride_), padding(padding_),
    bias(bias_){}
};
template <typename Fn>
class Conv2d : public iConv2d
{  
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
                    bool withgrad_=false)
        :iConv2d(inChannels_, h, w, outChannels_,
                 kernelSize_, stride_, padding_, bias_)
    {
        type = LAYER_CONV2D;
        /* (N, c, kernelSize, kernelSize) */
        Tensor &kernels = iLayer::w;
        kernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
        Random::uniform(kernels, -1, 1);
        ho = std::floor((hi - kernelSize + 2*padding)/stride) + 1;
        wo = std::floor((wi - kernelSize + 2*padding)/stride) + 1;
        /* (N, ho, wo) */
        o = Tensor(outChannels, ho, wo);
        if (bias == true) {
            /* (N, ho, wo) */
            b = Tensor(outChannels, kernelSize, kernelSize);
            Random::uniform(b, -1, 1);
        }

        if (withgrad_ == true) {
            g.w = Tensor(kernels.shape);
            g.b = Tensor(b.shape);
            v.w = Tensor(kernels.shape);
            v.b = Tensor(b.shape);
            m.w = Tensor(kernels.shape);
            m.b = Tensor(b.shape);

            e = Tensor(outChannels, ho, wo);
        }
    }
    static std::shared_ptr<Conv2d> _(int inChannels_,
                                     int h,
                                     int w,
                                     int outChannels_,
                                     int kernelSize_=3,
                                     int stride_=1,
                                     int padding_=0,
                                     bool bias_=false,
                                     bool withgrad_=true)
    {
        return std::make_shared<Conv2d>(inChannels_, h, w, outChannels_,
                                        kernelSize_, stride_, padding_,
                                        bias_, withgrad_);
    }

    Tensor& forward(const Tensor &x) override
    {
        /* conv */
        Tensor &kernels = iLayer::w;
        conv2d(o, kernels, x, stride, padding);
        /* bias */
        if (bias == true) {
            o += b;
        }
        /* activate */
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Fn::f(o[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor &kernels = iLayer::w;
        for (int n = 0; n < ei.shape[0]; n++) {
            for (int i = 0; i < ei.shape[1]; i++) {
                for (int j = 0; j < ei.shape[2]; j++) {

                    int h0 = (i - kernelSize + 1)/stride;
                    h0 = h0 > 0 ? std::ceil(h0):0;
                    int hn = i/stride;
                    hn = hn < ho ? std::floor(hn):ho;

                    int k0 = (j - kernelSize + 1)/stride;
                    k0 = k0 > 0 ? std::ceil(k0):0;
                    int kn = j/stride;
                    kn = kn < wo ? std::floor(kn):wo;

                    for (int c = 0; c < kernels.shape[0]; c++) {
                        for (int h = h0; h < hn; h++) {
                            for (int k = k0; k < kn; k++) {
                                ei(n, i, j) += kernels(n, c, i - h*stride, j - k*stride)*e(c, h, k);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    void gradient(const Tensor &x, const Tensor &y) override
    {
        Tensor dy(o.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Fn::d(o[i]) * e[i];
        }
        /* db */
        if (bias == true) {
            g.b += dy;
        }
        /* dkernel */
        Tensor &dkernels = g.w;
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
        o.zero();
        e.zero();
        return;
    }
};

class MaxPooling2d: public iConv2d
{
public:
    Tensor mask;
public:
    MaxPooling2d(){}
    explicit MaxPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        iConv2d(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        type = LAYER_MAXPOOLING;
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        mask = Tensor(outChannels, ho, wo);
        e = Tensor(outChannels, ho, wo);
    }
    static std::shared_ptr<MaxPooling2d> _(int inChannels_,
                                           int h,
                                           int w,
                                           int kernelSize_=2,
                                           int stride_=2)
    {
        return std::make_shared<MaxPooling2d>(inChannels_, h, w, kernelSize_, stride_);
    }

    Tensor& forward(const Tensor &x) override
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
    void backward(Tensor &ei) override
    {
        for (int n = 0; n < ei.shape[0]; n++) {
            for (int i = 0; i < ei.shape[1]; i++) {
                for (int j = 0; j < ei.shape[2]; j++) {

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
                            ei(n, i, j) += mask(n, h, k)*e(n, h, k);
                        }
                    }
                }
            }
        }
        mask.zero();
        return;
    }
    void gradient(const Tensor &x, const Tensor &) override{}
    void SGD(float learningRate) override {}
    void RMSProp(float rho, float learningRate, float decay) override {}
    void NormRMSProp(float rho, float learningRate, float decay) override {}
    void Adam(float alpha, float beta, float alpha_t, float beta_t,float learningRate, float decay) override{}
};

class AvgPooling2d: public iConv2d
{
public:
    AvgPooling2d(){}
    explicit AvgPooling2d(int inChannels_,
                          int h,
                          int w,
                          int kernelSize_=2,
                          int stride_=2):
        iConv2d(inChannels_, h, w, inChannels_, kernelSize_, stride_, 0, false)
    {
        type = LAYER_AVGPOOLING;
        ho = std::floor((hi - kernelSize)/stride) + 1;
        wo = std::floor((wi - kernelSize)/stride) + 1;
        o = Tensor(outChannels, ho, wo);
        e = Tensor(outChannels, ho, wo);
    }
    static std::shared_ptr<AvgPooling2d> _(int inChannels_,
                                           int h,
                                           int w,
                                           int kernelSize_=2,
                                           int stride_=2)
    {
        return std::make_shared<AvgPooling2d>(inChannels_, h, w, kernelSize_, stride_);
    }

    Tensor& forward(const Tensor &x) override
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

    void backward(Tensor &ei)
    {
        /* delta_: previous delta, the shape is same as delta and output */
        for (int n = 0; n < ei.shape[0]; n++) {
            for (int i = 0; i < ei.shape[1]; i++) {
                for (int j = 0; j < ei.shape[2]; j++) {

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
                            ei(n, i, j) += e(n, h, k);
                        }
                    }
                }
            }
        }
        return;
    }
    void gradient(const Tensor &x, const Tensor &) override{}
    void SGD(float learningRate) override {}
    void RMSProp(float rho, float learningRate, float decay) override {}
    void NormRMSProp(float rho, float learningRate, float decay) override {}
    void Adam(float alpha, float beta, float alpha_t, float beta_t,float learningRate, float decay) override{}
};

}
#endif // CONV2D_HPP
