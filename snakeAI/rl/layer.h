#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <memory>
#include <iostream>
#include "util.h"
#include "optimize.h"
#include "activate.h"
#include "loss.h"

namespace RL {

class Grad
{
public:
    Mat w;
    Mat b;
public:
    Grad(){}
    Grad(std::size_t outputDim, std::size_t inputDim)
    {
        w = Mat(outputDim, inputDim);
        b = Mat(outputDim, 1);
    }
    explicit Grad(const Grad &r)
        :w(r.w),b(r.b){}
    void zero()
    {
        w.zero();
        b.zero();
        return;
    }
};

class iLayer
{
public:
    std::size_t inputDim;
    std::size_t outputDim;
    Mat w;
    Mat b;
    Mat o;
    Mat e;
    Grad g;
    Grad v;
    Grad m;
public:
    iLayer(){}
    iLayer(std::size_t inputDim_, std::size_t outputDim_, bool trainFlag)
        :inputDim(inputDim_), outputDim(outputDim_)
    {
        w = Mat(outputDim, inputDim);
        b = Mat(outputDim, 1);
        o = Mat(outputDim, 1);
        e = Mat(outputDim, 1);
        if (trainFlag == true) {
            g = Grad(outputDim, inputDim);
            v = Grad(outputDim, inputDim);
            m = Grad(outputDim, inputDim);
        }
        uniformRand(w, -1, 1);
        uniformRand(b, -1, 1);
    }
    explicit iLayer(const iLayer &r)
        :inputDim(r.inputDim), outputDim(r.outputDim),
          w(r.w), b(r.b), o(r.o), e(r.e), g(r.g), v(r.v), m(r.m){}
    virtual ~iLayer(){}
    virtual Mat& forward(const Mat& x)
    {
        Mat::Mul::ikkj(o, w, x);
        o += b;
        return o;
    }

    virtual void gradient(const Mat& x, const Mat&)
    {
        Mat::Mul::ikjk(g.w, e, x);
        g.b += e;
        e.zero();
        o.zero();
        return;
    }

    virtual void backward(Mat &ei)
    {
        Mat::Mul::kikj(ei, w, e);
        return;
    }

    void SGD(float learningRate)
    {
        Optimize::SGD(w, g.w, learningRate);
        Optimize::SGD(b, g.b, learningRate);
        g.zero();
        return;
    }

    void RMSProp(float rho, float learningRate, float decay)
    {
        Optimize::RMSProp(w, v.w, g.w, learningRate, rho, decay);
        Optimize::RMSProp(b, v.b, g.b, learningRate, rho, decay);
        g.zero();
        return;
    }

    void Adam(float alpha, float beta, float alpha_t, float beta_t,float learningRate, float decay)
    {
        Optimize::Adam(w, v.w, m.w, g.w,
                        alpha_t, beta_t, learningRate, alpha, beta, decay);
        Optimize::Adam(b, v.b, m.b, g.b,
                        alpha_t, beta_t, learningRate, alpha, beta, decay);
        g.zero();
        return;
    }

};

template<typename Fn>
class Layer : public iLayer
{
public:
    Layer(){}
    virtual ~Layer(){}
    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool tarinFlag = true)
    {
        return std::make_shared<Layer>(inputDim, outputDim, tarinFlag);
    }

    Layer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iLayer(inputDim, outputDim, trainFlag){}

    Mat& forward(const Mat& x) override
    {
        Mat::Mul::ikkj(o, w, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Fn::f(o[i]);
        }
        return o;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Fn::d(o[i]) * e[i];
        }
        Mat::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }
};

class SoftmaxLayer : public iLayer
{
public:
    SoftmaxLayer(){}
    ~SoftmaxLayer(){}
    explicit SoftmaxLayer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iLayer(inputDim, outputDim, trainFlag){}

    static std::shared_ptr<SoftmaxLayer> _(std::size_t inputDim,
                                           std::size_t outputDim,
                                           bool tarinFlag)
    {
        return std::make_shared<SoftmaxLayer>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Mat::Mul::ikkj(o, w, x);
        o += b;
        float s = 0;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            s += std::exp(o[i]);
        }
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = std::exp(o[i]) / s;
        }
        return o;
    }

    void gradient(const RL::Mat &x, const Mat &y) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = o[i] - y[i];
        }
        Mat::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class GeluLayer : public iLayer
{
public:
    Mat op;
public:
    GeluLayer(){}
    ~GeluLayer(){}
    explicit GeluLayer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iLayer(inputDim, outputDim, trainFlag),op(outputDim, 1){}

    static std::shared_ptr<GeluLayer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool tarinFlag)
    {
        return std::make_shared<GeluLayer>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Mat::Mul::ikkj(op, w, x);
        op += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Gelu::f(op[i]);
        }
        return o;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Gelu::d(op[i]) * e[i];
        }
        Mat::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class SwishLayer : public iLayer
{
public:
    Mat op;
public:
    SwishLayer(){}
    ~SwishLayer(){}
    explicit SwishLayer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iLayer(inputDim, outputDim, trainFlag),op(outputDim, 1){}

    static std::shared_ptr<SwishLayer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SwishLayer>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Mat::Mul::ikkj(op, w, x);
        op += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Swish::f(op[i]);
        }
        return o;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Swish::d(op[i]) * e[i];
        }
        Mat::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

template<typename Fn>
class Dropout : public Layer<Fn>
{
public:
    bool trainFlag;
    float p;
    Mat mask;
public:
    Dropout(){}
    ~Dropout(){}
    explicit Dropout(std::size_t inputDim, std::size_t outputDim,
                          bool trainFlag_, float p_)
        :Layer<Fn>(inputDim, outputDim, trainFlag_),
          trainFlag(trainFlag_), p(p_), mask(outputDim, 1){}

    static std::shared_ptr<Dropout> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool tarinFlag, float p_)
    {
        return std::make_shared<Dropout>(inputDim, outputDim, tarinFlag, p_);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Layer<Fn>::forward(x);
        if (trainFlag == true) {
            std::bernoulli_distribution bernoulli(p);
            for (std::size_t i = 0; i < Layer<Fn>::o.size(); i++) {
                mask[i] = bernoulli(Rand::engine) / (1 - p);
            }
            Layer<Fn>::o *= mask;
        }
        return Layer<Fn>::o;
    }

    void backward(Mat& ei) override
    {
        if (trainFlag == true) {
            Layer<Fn>::e *= mask;
        }
        Layer<Fn>::backward(ei);
        return;
    }
};


template<typename Fn>
class LayerNorm : public iLayer
{
public:
    float gamma;
    float u;
    Mat op;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Mat(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Mat::Mul::ikkj(op, w, x);
        u = op.mean();
        float sigma = op.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(gamma*(op[i] - u) + b[i]);
        }
        return o;
    }

    void backward(Mat &ei) override
    {
        Mat::Mul::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float error = Fn::d(o[i])*e[i];
#if 1
            float gamma3 = gamma*gamma*gamma;
            float delta = op[i] - u;
            dy[i] = (1.0 - 1.0/float(outputDim))*(gamma - delta*delta*gamma3)*error;
#else
            dy[i] = gamma*Fn::d(o[i])*e[i];
#endif
            g.b[i] += error;
        }
        Mat::Mul::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class PreNorm : public iLayer
{
public:
    float gamma;
    float u;
    Mat x_;
    Mat op;
public:
    PreNorm(){}
    ~PreNorm(){}
    explicit PreNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        x_ = Mat(inputDim, 1);
        op = Mat(outputDim, 1);
    }

    static std::shared_ptr<PreNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<PreNorm>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        u = x.mean();
        float sigma = x.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < x.size(); i++) {
            x_[i] = gamma*(x[i] - u);
        }
        Mat::Mul::ikkj(op, w, x_);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(op[i] + b[i]);
        }
        return o;
    }

    void backward(Mat &ei) override
    {
        Mat::Mul::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Fn::d(o[i])*e[i];
        }
        Mat dx(inputDim, 1);
        for (std::size_t i = 0; i < dx.totalSize; i++) {
            float delta = (x[i] - u)*gamma;
            dx[i] = (1 - 1.0/float(inputDim))*(1 - delta*delta)*gamma*x_[i];
        }
        Mat::Mul::ikjk(g.w, dy, dx);
        g.b += dy;
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class RMSNorm : public iLayer
{
public:
    float gamma;
    Mat op;
public:
    RMSNorm(){}
    ~RMSNorm(){}
    explicit RMSNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Mat(outputDim, 1);
    }

    static std::shared_ptr<RMSNorm> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool tarinFlag)
    {
        return std::make_shared<RMSNorm>(inputDim, outputDim, tarinFlag);
    }
    Mat& forward(const RL::Mat &x) override
    {
        Mat::Mul::ikkj(op, w, x);
        float sigma = op.variance(0);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(gamma*op[i] + b[i]);
        }
        return o;
    }

    void backward(Mat &ei) override
    {
        Mat::Mul::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        Mat dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float error = Fn::d(o[i])*e[i];
            float gamma3 = gamma*gamma*gamma;
            dy[i] = (gamma - op[i]*op[i]*gamma3)*error;
            g.b[i] += error;
        }
        Mat::Mul::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }
};

}
#endif // LAYER_H
