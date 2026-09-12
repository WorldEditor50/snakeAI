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
    /*
       Flat strides instead of Tensor::operator(). operator() calls posOf(),
       which builds an int index array and loops over it on EVERY element
       access, and this inner loop performs three such accesses per
       multiply-accumulate. That dominated ConvDQN: its rollout performs ~160
       network forwards per game step and the convolution was costing roughly
       10 ns per MAC. The input-validity bounds are now hoisted out of the
       innermost loops as explicit ranges. The arithmetic is unchanged.
    */
    const int outC = y.shape[0];
    const int ho   = y.shape[1];
    const int wo   = y.shape[2];
    const int inC  = kernels.shape[1];
    const int kH   = kernels.shape[2];
    const int kW   = kernels.shape[3];
    const int hi   = x.shape[1];
    const int wi   = x.shape[2];

    const float *xData = x.val.data();
    const float *kData = kernels.val.data();
    float *yData = y.val.data();

    const int xStrideC = hi*wi;
    const int xStrideH = wi;
    const int kStrideO = inC*kH*kW;
    const int kStrideC = kH*kW;
    const int kStrideH = kW;
    const int yStrideO = ho*wo;
    const int yStrideH = wo;

    for (int oc = 0; oc < outC; oc++) {
        const float *koc = kData + oc*kStrideO;
        float *yoc = yData + oc*yStrideO;
        for (int i = 0; i < ho; i++) {
            /* kernel rows that land inside x for this output row */
            int uBeg = padding - i*stride;
            if (uBeg < 0) { uBeg = 0; }
            int uEnd = hi + padding - i*stride;
            if (uEnd > kH) { uEnd = kH; }
            const int xRowBase = i*stride - padding;
            for (int j = 0; j < wo; j++) {
                int vBeg = padding - j*stride;
                if (vBeg < 0) { vBeg = 0; }
                int vEnd = wi + padding - j*stride;
                if (vEnd > kW) { vEnd = kW; }
                const int xColBase = j*stride - padding;
                float acc = 0;
                for (int ic = 0; ic < inC; ic++) {
                    const float *xic = xData + ic*xStrideC;
                    const float *kic = koc + ic*kStrideC;
                    for (int u = uBeg; u < uEnd; u++) {
                        const float *xrow = xic + (xRowBase + u)*xStrideH;
                        const float *krow = kic + u*kStrideH;
                        for (int v = vBeg; v < vEnd; v++) {
                            acc += krow[v]*xrow[xColBase + v];
                        }
                    }
                }
                yoc[j] = acc;
            }
            yoc += yStrideH;
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
        /*
         * Derive the spatial geometry from the ACTUAL input instead of the
         * height/width handed to the constructor, which are only a hint.
         * When they disagree with the real input the convolution used to
         * silently produce a CROPPED result (and read past the end of x when the
         * input was smaller). `hi`/`wi` are also what backward() uses for its
         * bounds checks and for the kernel-gradient loop, so keeping them in
         * sync with the data is what keeps backward correct.
         * For the existing networks the declarations already match the data, so
         * this is a no-op there.
         */
        if (x.shape.size() >= 3) {
            inChannels = x.shape[0];
            hi = x.shape[1];
            wi = x.shape[2];
        }
        ho = (hi - kernelSize + 2*padding)/stride + 1;
        wo = (wi - kernelSize + 2*padding)/stride + 1;
        if (ho < 1) { ho = 1; }
        if (wo < 1) { wo = 1; }
        op = Tensor(outChannels, ho, wo);
        o = Tensor(outChannels, ho, wo);
        e = Tensor(outChannels, ho, wo);
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
        /* dz = dL/d(pre-activation) = tanh'(o) ⊙ e. It must be computed BEFORE
           the input gradient, because ei is the gradient through the
           convolution of dz. The previous version used the raw e for ei, which
           silently dropped the tanh' factor from the input gradient. */
        Tensor dy(o.shape);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Tanh::df(o[i])*e[i];
        }
        ei.zero();
        /* Channel count comes from `o` (the layer's shaped output), not from
           `e.shape[0]`: a flat (totalSize,1) error tensor — which is what
           Loss::MSE::df() returns — would otherwise be mistaken for that many
           channels and index `e` out of bounds. */
        for (int oc = 0; oc < o.shape[0]; oc++) {
            for (int h_out = 0; h_out < ho; h_out++) {
                for (int w_out = 0; w_out < wo; w_out++) {
                    float dy_val = dy(oc, h_out, w_out);
                    if (dy_val == 0) continue;
                    for (int ic = 0; ic < kernels.shape[1]; ic++) {
                        for (int u = 0; u < kernelSize; u++) {
                            for (int v = 0; v < kernelSize; v++) {
                                int hi_idx = u + h_out*stride - padding;
                                int wi_idx = v + w_out*stride - padding;
                                if (hi_idx < 0 || hi_idx >= ei.shape[1] ||
                                    wi_idx < 0 || wi_idx >= ei.shape[2]) {
                                    continue;
                                }
                                ei(ic, hi_idx, wi_idx) += kernels(oc, ic, u, v) * dy_val;
                            }
                        }
                    }
                }
            }
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
        /* Optimize::SGD(w, dw, lr, gamma, clipGrad): the 4th argument is gamma
           (weight decay), NOT clipGrad. Passing `true` there set gamma=1.0 and
           replaced the weights with -lr*dw every step. */
        Optimize::SGD(kernels, g.kernels, lr);
        if (bias) {
            Optimize::SGD(b, g.b, lr);
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
        /* Output geometry comes from the ACTUAL input, not the declared h/w —
           see the note in Conv2d::forward. */
        if (x.shape.size() >= 3) {
            outChannels = x.shape[0];
            hi = x.shape[1];
            wi = x.shape[2];
        }
        ho = (hi - kernelSize)/stride + 1;
        wo = (wi - kernelSize)/stride + 1;
        if (ho < 1) { ho = 1; }
        if (wo < 1) { wo = 1; }
        o = Tensor(outChannels, ho, wo);
        mask = Tensor(outChannels, ho, wo);
        e = Tensor(outChannels, ho, wo);
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
        /* Output geometry comes from the ACTUAL input — see Conv2d::forward. */
        if (x.shape.size() >= 3) {
            outChannels = x.shape[0];
            hi = x.shape[1];
            wi = x.shape[2];
        }
        ho = (hi - kernelSize)/stride + 1;
        wo = (wi - kernelSize)/stride + 1;
        if (ho < 1) { ho = 1; }
        if (wo < 1) { wo = 1; }
        o = Tensor(outChannels, ho, wo);
        e = Tensor(outChannels, ho, wo);
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
