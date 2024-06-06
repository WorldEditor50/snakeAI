#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"

namespace RL {

class iFcLayer : public iLayer
{
public:
    class FcGrad
    {
    public:
        Tensor w;
        Tensor b;
    public:
        FcGrad(){}
        FcGrad(std::size_t outputDim, std::size_t inputDim)
        {
            w = Tensor(outputDim, inputDim);
            b = Tensor(outputDim, 1);
        }
        explicit FcGrad(const FcGrad &r)
            :w(r.w),b(r.b){}
        void zero()
        {
            w.zero();
            b.zero();
            return;
        }
    };
public:
    std::size_t inputDim;
    std::size_t outputDim;
    Tensor w;
    Tensor b;
    FcGrad g;
    FcGrad v;
    FcGrad m;
public:
    iFcLayer(){}
    iFcLayer(std::size_t inputDim_, std::size_t outputDim_, bool trainFlag)
        :inputDim(inputDim_), outputDim(outputDim_)
    {
        type = iLayer::LAYER_FC;
        w = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (trainFlag == true) {
            g = FcGrad(outputDim, inputDim);
            v = FcGrad(outputDim, inputDim);
            m = FcGrad(outputDim, inputDim);
        }
        Random::uniform(w, -1, 1);
        Random::uniform(b, -1, 1);
    }
    explicit iFcLayer(const iFcLayer &r)
        :inputDim(r.inputDim), outputDim(r.outputDim){}
    virtual ~iFcLayer(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        Tensor::MM::ikkj(o, w, x);
        o += b;
        return o;
    }

    virtual void gradient(const Tensor& x, const Tensor&)
    {
        Tensor::MM::ikjk(g.w, e, x);
        g.b += e;
        e.zero();
        o.zero();
        return;
    }

    virtual void backward(Tensor &ei)
    {
        Tensor::MM::kikj(ei, w, e);
        return;
    }
    virtual void SGD(float learningRate) override
    {
        Optimize::SGD(w, g.w, learningRate, true);
        Optimize::SGD(b, g.b, learningRate, true);
        g.zero();
        return;
    }

    virtual void RMSProp(float rho, float lr, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w, v.w, g.w, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        g.zero();
        return;
    }

    virtual void Adam(float alpha, float beta,
                      float alpha_, float beta_,
                      float lr, float decay, bool clipGrad) override
    {
        Optimize::Adam(w, v.w, m.w, g.w,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        g.zero();
        return;
    }
    virtual void clamp(float c0, float cn) override
    {
        Optimize::clamp(w, c0, cn);
        Optimize::clamp(b, c0, cn);
        return;
    }
    virtual void copyTo(iLayer* layer) override
    {
        iFcLayer *pLayer = static_cast<iFcLayer*>(layer);
        pLayer->w = w;
        pLayer->b = b;
        return;
    }
    virtual void softUpdateTo(iLayer* layer, float alpha) override
    {
        iFcLayer *pLayer = static_cast<iFcLayer*>(layer);
        lerp(pLayer->w, w, alpha);
        lerp(pLayer->b, b, alpha);
        return;
    }
};

template<typename Fn>
class Layer : public iFcLayer
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
        :iFcLayer(inputDim, outputDim, trainFlag){}

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        Tensor::MM::ikkj(o, w, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Fn::f(o[i]);
        }
        return o;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Fn::d(o[i]) * e[i];
        }
        Tensor::MM::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }
};

class Softmax : public iFcLayer
{
public:
    Softmax() {}
    ~Softmax(){}
    explicit Softmax(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iFcLayer(inputDim, outputDim, trainFlag)
    {

    }

    static std::shared_ptr<Softmax> _(std::size_t inputDim, std::size_t outputDim, bool tarinFlag)
    {
        return std::make_shared<Softmax>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o, w, x);
        o += b;
        return RL::softmax(o);
    }

    void gradient(const RL::Tensor &x, const Tensor &y) override
    {
        Tensor dy = o - y;
        Tensor::MM::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class GeluLayer : public iFcLayer
{
public:
    Tensor op;
public:
    GeluLayer(){}
    ~GeluLayer(){}
    explicit GeluLayer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iFcLayer(inputDim, outputDim, trainFlag),op(outputDim, 1){}

    static std::shared_ptr<GeluLayer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool tarinFlag)
    {
        return std::make_shared<GeluLayer>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        op += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Gelu::f(op[i]);
        }
        return o;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Gelu::d(op[i]) * e[i];
        }
        Tensor::MM::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class SwishLayer : public iFcLayer
{
public:
    Tensor op;
public:
    SwishLayer(){}
    ~SwishLayer(){}
    explicit SwishLayer(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iFcLayer(inputDim, outputDim, trainFlag),op(outputDim, 1){}

    static std::shared_ptr<SwishLayer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SwishLayer>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        op += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Swish::f(op[i]);
        }
        return o;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Swish::d(op[i]) * e[i];
        }
        Tensor::MM::ikjk(g.w, dy, x);
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
    Tensor mask;
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
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Layer<Fn>::forward(x);
        if (trainFlag == true) {
            std::bernoulli_distribution bernoulli(p);
            for (std::size_t i = 0; i < Layer<Fn>::o.size(); i++) {
                mask[i] = bernoulli(Random::engine) / (1 - p);
            }
            Layer<Fn>::o *= mask;
        }
        return Layer<Fn>::o;
    }

    void backward(Tensor& ei) override
    {
        if (trainFlag == true) {
            Layer<Fn>::e *= mask;
        }
        Layer<Fn>::backward(ei);
        return;
    }
};

namespace LN {
struct Def{};
struct Pre{};
struct Post{};
};
template<typename Fn, typename Type=LN::Def>
class LayerNorm{};

template<typename Fn>
class LayerNorm<Fn, LN::Def> : public iFcLayer
{
public:
    float gamma;
    float u;
    Tensor op;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iFcLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        u = op.mean();
        float sigma = op.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f((op[i] - u)*gamma + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float error = Fn::d(o[i])*e[i];
            float d = (op[i] - u)*gamma;
            dy[i] = (1.0 - 1.0/float(outputDim))*(1 - d*d)*gamma*error;
            g.b[i] += error;
        }
        Tensor::MM::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class LayerNorm<Fn, LN::Pre> : public iFcLayer
{
public:
    float gamma;
    float u;
    Tensor x_;
    Tensor op;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iFcLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        x_ = Tensor(inputDim, 1);
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        u = x.mean();
        float sigma = x.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < x.size(); i++) {
            x_[i] = (x[i] - u)*gamma;
        }
        Tensor::MM::ikkj(op, w, x_);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(op[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            dy[i] = Fn::d(o[i])*e[i];
        }
        Tensor dx(inputDim, 1);
        for (std::size_t i = 0; i < dx.totalSize; i++) {
            //float d = (x[i] - u)*gamma;
            float d = x_[i];
            dx[i] = (1 - 1.0/float(inputDim))*(1 - d*d)*gamma*d;
        }
        Tensor::MM::ikjk(g.w, dy, dx);
        g.b += dy;
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class LayerNorm<Fn, LN::Post> : public iFcLayer
{
public:
    float gamma;
    float u;
    Tensor o1;
    Tensor o2;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iFcLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        o1 = Tensor(outputDim, 1);
        o2 = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o1, w, x);
        for (std::size_t i = 0; i < o.size(); i++) {
            o2[i] = Fn::f(o1[i] + b[i]);
        }
        u = o2.mean();
        float sigma = o2.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = (o2[i] - u)*gamma;
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float d = o2[i];//(o2[i] - u)*gamma;
            dy[i] = (1 - d*d)*(1 - 1.0/float(outputDim))*Fn::d(o2[i])*gamma*e[i];
        }
        Tensor::MM::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o1.zero();
        return;
    }

};


template<typename Fn>
class RMSNorm : public iFcLayer
{
public:
    float gamma;
    Tensor op;
public:
    RMSNorm(){}
    ~RMSNorm(){}
    explicit RMSNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iFcLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<RMSNorm> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool tarinFlag)
    {
        return std::make_shared<RMSNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        float sigma = op.variance(0);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(gamma*op[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float error = Fn::d(o[i])*e[i];
            float d = op[i]*gamma;
            dy[i] = (1 - d*d)*gamma*error;
            g.b[i] += error;
        }
        Tensor::MM::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }
};


template<typename Fn>
class TanhNorm : public iFcLayer
{
public:
    Tensor o1;
    Tensor o2;
public:
    TanhNorm(){}
    ~TanhNorm(){}
    explicit TanhNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iFcLayer(inputDim, outputDim, trainFlag_)
    {
        o1 = Tensor(outputDim, 1);
        o2 = Tensor(outputDim, 1);
    }

    static std::shared_ptr<TanhNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<TanhNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o1, w, x);
        o2 = RL::tanh(o1);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(o2[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float error = Fn::d(o[i])*e[i];
            float d = o2[i];
            dy[i] = (1 - d*d)*error;
            g.b[i] += error;
        }
        Tensor::MM::ikjk(g.w, dy, x);
        e.zero();
        o1.zero();
        return;
    }

};

}
#endif // LAYER_H
