#ifndef CONV2D_HPP
#define CONV2D_HPP
#include <memory>
#include "tensor.hpp"
#include "activate.h"
#include "optimize.h"
#include "util.hpp"
#include "ilayer.h"

namespace RL {

inline void conv2d(Tensor &y, const Tensor &kernels, const Tensor &x, int stride=1, int padding=0)
{
    /* output shape: (outChannels, ho, wo) */
    /* kernels shape: (outChannels, inChannels, kernelSize, kernelSize) */
    /* x shape: (inChannels, hi, wi) */
    for (int oc = 0; oc < y.shape[0]; oc++) {
        for (int i = 0; i < y.shape[1]; i++) {
            for (int j = 0; j < y.shape[2]; j++) {
                float ynij = 0;
                for (int ic = 0; ic < kernels.shape[1]; ic++) {
                    for (int u = 0; u < kernels.shape[2]; u++) {
                        for (int v = 0; v < kernels.shape[3]; v++) {
                            /* map to input  */
                            int ui = u + i*stride - padding;
                            int vj = v + j*stride - padding;
                            if (ui < 0 || ui >= x.shape[1] ||
                                    vj < 0 || vj >= x.shape[2]) {
                                continue;
                            }
                            /* sum up all convolution result */
                            ynij += kernels(oc, ic, u, v)*x(ic, ui, vj);
                        }
                    }
                }
                y(oc, i, j) = ynij;
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
    class Conv2dGrad
    {
    public:
        Tensor kernels;
        Tensor b;
    public:
        Conv2dGrad(){}
        void zero()
        {
            kernels.zero();
            b.zero();
            return;
        }
    };
public:
    Tensor kernels;
    Tensor b;
    Tensor op;
    Conv2dGrad g;
    Conv2dGrad v;
    Conv2dGrad m;
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
        kernels = Tensor(outChannels, inChannels, kernelSize, kernelSize);
        Random::uniform(kernels, -1, 1);
        ho = std::floor((hi - kernelSize + 2*padding)/stride) + 1;
        wo = std::floor((wi - kernelSize + 2*padding)/stride) + 1;
        /* (N, ho, wo) */
        o = Tensor(outChannels, ho, wo);
        op = Tensor(outChannels, ho, wo);
        if (bias == true) {
            /* (outChannels, 1, 1) - one bias per output channel */
            b = Tensor(outChannels, 1, 1);
            Random::uniform(b, -1, 1);
        }

        if (withgrad_ == true) {
            g.kernels = Tensor(kernels.shape);
            g.b = Tensor(b.shape);
            v.kernels = Tensor(kernels.shape);
            v.b = Tensor(b.shape);
            m.kernels = Tensor(kernels.shape);
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

    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        /* conv */
        conv2d(op, kernels, x, stride, padding);
        /* bias - manually broadcast (outChannels, 1, 1) to (outChannels, ho, wo) */
        if (bias) {
            for (int oc = 0; oc < outChannels; oc++) {
                float bval = b(oc, 0, 0);
                for (int i = 0; i < ho; i++) {
                    for (int j = 0; j < wo; j++) {
                        op(oc, i, j) += bval;
                    }
                }
            }
        }
        /* activate */
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(op[i]);
        }
        return o;
    }

    void backward(const Tensor &x, Tensor &ei) override
    {
        /* ei shape: (inChannels, hi, wi) - gradient flowing back to input */
        /* kernels shape: (outChannels, inChannels, kernelSize, kernelSize) */
        /* e shape: (outChannels, ho, wo) - error from output */
        ei.zero();
        for (int oc = 0; oc < e.shape[0]; oc++) {
            for (int h_out = 0; h_out < ho; h_out++) {
                for (int w_out = 0; w_out < wo; w_out++) {
                    float e_val = e(oc, h_out, w_out);
                    if (e_val == 0) continue;
                    for (int ic = 0; ic < kernels.shape[1]; ic++) {
                        for (int u = 0; u < kernelSize; u++) {
                            for (int v = 0; v < kernelSize; v++) {
                                int hi_idx = u + h_out*stride - padding;
                                int wi_idx = v + w_out*stride - padding;
                                if (hi_idx < 0 || hi_idx >= ei.shape[1] ||
                                    wi_idx < 0 || wi_idx >= ei.shape[2]) {
                                    continue;
                                }
                                ei(ic, hi_idx, wi_idx) += kernels(oc, ic, u, v) * e_val;
                            }
                        }
                    }
                }
            }
        }

        Tensor dy(o.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Tanh::df(o[i])*e[i];
        }
        /* db: gradient for bias, sum dy over spatial dimensions */
        if (bias) {
            for (int oc = 0; oc < outChannels; oc++) {
                float sum = 0;
                for (int i = 0; i < ho; i++) {
                    for (int j = 0; j < wo; j++) {
                        sum += dy(oc, i, j);
                    }
                }
                g.b(oc, 0, 0) += sum;
            }
        }
        /* dkernel */
        /* x shape: (inChannels, hi, wi) */
        /* dy shape: (outChannels, ho, wo) */
        /* dkernels shape: (outChannels, inChannels, kernelSize, kernelSize) */
        Tensor &dkernels = g.kernels;
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int h_out = 0; h_out < ho; h_out++) {
                    for (int w_out = 0; w_out < wo; w_out++) {
                        float dy_val = dy(oc, h_out, w_out);
                        if (dy_val == 0) continue;
                        for (int u = 0; u < kernelSize; u++) {
                            for (int v = 0; v < kernelSize; v++) {
                                int hi_idx = u + h_out*stride - padding;
                                int wi_idx = v + w_out*stride - padding;
                                if (hi_idx < 0 || hi_idx >= hi ||
                                    wi_idx < 0 || wi_idx >= wi) {
                                    continue;
                                }
                                dkernels(oc, ic, u, v) += x(ic, hi_idx, wi_idx) * dy_val;
                            }
                        }
                    }
                }
            }
        }
        op.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(kernels, g.kernels, lr, true);
        if (bias) {
            Optimize::SGD(b, g.b, lr, true);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(kernels, v.kernels, g.kernels, lr, rho, decay, clipGrad);
        if (bias) {
            Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(kernels, v.kernels, m.kernels, g.kernels,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        if (bias) {
            Optimize::Adam(b, v.b, m.b, g.b,
                           alpha_, beta_, lr,
                           alpha, beta, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(kernels, c0, cn);
        if (bias) {
            Optimize::clamp(b, c0, cn);
        }
        return;
    }

    virtual void copyTo(iLayer* layer) override
    {
        Conv2d *pLayer = static_cast<Conv2d*>(layer);
        pLayer->kernels = kernels;
        if (bias) {
            pLayer->b = b;
        }
        return;
    }
    virtual void softUpdateTo(iLayer* layer, float alpha) override
    {
        Conv2d *pLayer = static_cast<Conv2d*>(layer);
        lerp(pLayer->kernels, kernels, alpha);
        if (bias) {
            lerp(pLayer->b, b, alpha);
        }
        return;
    }

    virtual void write(std::ofstream &file) override
    {
        /* kernels */
        file<<kernels.toString()<<std::endl;
        /* b */
        file<<b.toString()<<std::endl;
        return;
    }

    virtual void read(std::ifstream &file) override
    {
        /* kernels */
        std::string ws;
        std::getline(file, ws);
        kernels = Tensor::fromString(ws);
        /* b */
        std::string bs;
        std::getline(file, bs);
        b = Tensor::fromString(bs);
        return;
    }
};

class MaxPooling2d: public iConv2d
{
public:
    /* mask stores per-output-position the kernel offset of the max value
       encoded as: h_offset * kernelSize + k_offset */
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
        /* mask stores per-output position encoded index of max */
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

    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        o.zero();
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    float maxValue = -1e10f;
                    int maxIdx = 0;
                    for (int h = 0; h < kernelSize; h++) {
                        for (int k = 0; k < kernelSize; k++) {
                            float value = x(n, h + i*stride, k + j*stride);
                            if (value > maxValue) {
                                maxValue = value;
                                maxIdx = h * kernelSize + k;
                            }
                        }
                    }
                    o(n, i, j) = maxValue;
                    mask(n, i, j) = static_cast<float>(maxIdx);
                }
            }
        }
        return o;
    }
    void backward(const Tensor &x, Tensor &ei) override
    {
        /* ei shape: (inChannels, hi, wi) - gradient to previous layer */
        /* For max pooling, gradient goes only to the input position that had the max value */
        ei.zero();
        for (int n = 0; n < outChannels; n++) {
            for (int i = 0; i < ho; i++) {
                for (int j = 0; j < wo; j++) {
                    int encoded = static_cast<int>(mask(n, i, j));
                    int h_offset = encoded / kernelSize;
                    int k_offset = encoded % kernelSize;
                    int hi_idx = h_offset + i * stride;
                    int wi_idx = k_offset + j * stride;
                    if (hi_idx >= 0 && hi_idx < hi && wi_idx >= 0 && wi_idx < wi) {
                        ei(n, hi_idx, wi_idx) += e(n, i, j);
                    }
                }
            }
        }
        e.zero();
        return;
    }

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

    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        /* conv */
        o.zero();
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

    void backward(const Tensor &x, Tensor &ei) override
    {
        /* Average pooling backward: evenly distribute output gradient
           to all input positions in the pooling window.
           ei shape: (inChannels, hi, wi) */
        float scale = 1.0f / static_cast<float>(kernelSize * kernelSize);
        ei.zero();
        for (int n = 0; n < outChannels; n++) {
            for (int h_out = 0; h_out < ho; h_out++) {
                for (int w_out = 0; w_out < wo; w_out++) {
                    float e_val = e(n, h_out, w_out) * scale;
                    if (e_val == 0) continue;
                    for (int u = 0; u < kernelSize; u++) {
                        for (int v = 0; v < kernelSize; v++) {
                            int hi_idx = u + h_out * stride;
                            int wi_idx = v + w_out * stride;
                            if (hi_idx < hi && wi_idx < wi) {
                                ei(n, hi_idx, wi_idx) += e_val;
                            }
                        }
                    }
                }
            }
        }
        e.zero();
        return;
    }

};

}
#endif // CONV2D_HPP
