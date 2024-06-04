#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "loss.h"

namespace RL {

class Grad
{
public:
    Tensor w;
    Tensor b;
public:
    Grad(){}
    Grad(std::size_t outputDim, std::size_t inputDim)
    {
        w = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
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
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING
    };
public:
    int type;
    std::size_t inputDim;
    std::size_t outputDim;
    Tensor w;
    Tensor b;
    Tensor o;
    Tensor e;
    Grad g;
    Grad v;
    Grad m;
public:
    iLayer(){}
    iLayer(std::size_t inputDim_, std::size_t outputDim_, bool trainFlag)
        :inputDim(inputDim_), outputDim(outputDim_)
    {
        type = LAYER_FC;
        w = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (trainFlag == true) {
            g = Grad(outputDim, inputDim);
            v = Grad(outputDim, inputDim);
            m = Grad(outputDim, inputDim);
        }
        Random::uniform(w, -1, 1);
        Random::uniform(b, -1, 1);
    }
    explicit iLayer(const iLayer &r)
        :type(r.type), inputDim(r.inputDim), outputDim(r.outputDim),
          w(r.w), b(r.b), o(r.o), e(r.e), g(r.g), v(r.v), m(r.m){}
    virtual ~iLayer(){}
    virtual Tensor& forward(const Tensor& x)
    {
        Tensor::Mul::ikkj(o, w, x);
        o += b;
        return o;
    }

    virtual void gradient(const Tensor& x, const Tensor&)
    {
        Tensor::Mul::ikjk(g.w, e, x);
        g.b += e;
        e.zero();
        o.zero();
        return;
    }

    virtual void backward(Tensor &ei)
    {
        Tensor::Mul::kikj(ei, w, e);
        return;
    }

    virtual void SGD(float learningRate)
    {
        Optimize::SGD(w, g.w, learningRate);
        Optimize::SGD(b, g.b, learningRate);
        g.zero();
        return;
    }

    virtual void RMSProp(float rho, float learningRate, float decay)
    {
        Optimize::RMSProp(w, v.w, g.w, learningRate, rho, decay);
        Optimize::RMSProp(b, v.b, g.b, learningRate, rho, decay);
        g.zero();
        return;
    }

    virtual void NormRMSProp(float rho, float learningRate, float decay)
    {
        Optimize::NormRMSProp(w, v.w, g.w, learningRate, rho, decay);
        Optimize::NormRMSProp(b, v.b, g.b, learningRate, rho, decay);
        g.zero();
        return;
    }

    virtual void Adam(float alpha, float beta, float alpha_t, float beta_t,float learningRate, float decay)
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

    Tensor& forward(const Tensor& x) override
    {
        Tensor::Mul::ikkj(o, w, x);
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
        Tensor::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }
};

class Softmax : public iLayer
{
public:
    Softmax() {}
    ~Softmax(){}
    explicit Softmax(std::size_t inputDim, std::size_t outputDim, bool trainFlag)
        :iLayer(inputDim, outputDim, trainFlag)
    {

    }

    static std::shared_ptr<Softmax> _(std::size_t inputDim, std::size_t outputDim, bool tarinFlag)
    {
        return std::make_shared<Softmax>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(o, w, x);
        o += b;
        return RL::softmax(o);
    }

    void gradient(const RL::Tensor &x, const Tensor &y) override
    {
        Tensor dy = o - y;
        Tensor::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class GeluLayer : public iLayer
{
public:
    Tensor op;
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
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(op, w, x);
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
        Tensor::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o.zero();
        return;
    }

};

class SwishLayer : public iLayer
{
public:
    Tensor op;
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
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(op, w, x);
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
        Tensor::Mul::ikjk(g.w, dy, x);
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
    Tensor& forward(const RL::Tensor &x) override
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
class LayerNorm<Fn, LN::Def> : public iLayer
{
public:
    float gamma;
    float u;
    Tensor op;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(op, w, x);
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
        Tensor::Mul::kikj(ei, w, e*gamma);
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
        Tensor::Mul::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class LayerNorm<Fn, LN::Pre> : public iLayer
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
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
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
    Tensor& forward(const RL::Tensor &x) override
    {
        u = x.mean();
        float sigma = x.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < x.size(); i++) {
            x_[i] = (x[i] - u)*gamma;
        }
        Tensor::Mul::ikkj(op, w, x_);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(op[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::Mul::kikj(ei, w, e*gamma);
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
        Tensor::Mul::ikjk(g.w, dy, dx);
        g.b += dy;
        e.zero();
        op.zero();
        return;
    }

};

template<typename Fn>
class LayerNorm<Fn, LN::Post> : public iLayer
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
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
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
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(o1, w, x);
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
        Tensor::Mul::kikj(ei, w, e*gamma);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < dy.totalSize; i++) {
            float d = o2[i];//(o2[i] - u)*gamma;
            dy[i] = (1 - d*d)*(1 - 1.0/float(outputDim))*Fn::d(o2[i])*gamma*e[i];
        }
        Tensor::Mul::ikjk(g.w, dy, x);
        g.b += dy;
        e.zero();
        o1.zero();
        return;
    }

};


template<typename Fn>
class RMSNorm : public iLayer
{
public:
    float gamma;
    Tensor op;
public:
    RMSNorm(){}
    ~RMSNorm(){}
    explicit RMSNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<RMSNorm> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool tarinFlag)
    {
        return std::make_shared<RMSNorm>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(op, w, x);
        float sigma = op.variance(0);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(gamma*op[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::Mul::kikj(ei, w, e*gamma);
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
        Tensor::Mul::ikjk(g.w, dy, x);
        e.zero();
        op.zero();
        return;
    }
};


template<typename Fn>
class TanhNorm : public iLayer
{
public:
    Tensor o1;
    Tensor o2;
public:
    TanhNorm(){}
    ~TanhNorm(){}
    explicit TanhNorm(std::size_t inputDim, std::size_t outputDim, bool trainFlag_)
        :iLayer(inputDim, outputDim, trainFlag_)
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
    Tensor& forward(const RL::Tensor &x) override
    {
        Tensor::Mul::ikkj(o1, w, x);
        o2 = RL::tanh(o1);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = Fn::f(o2[i] + b[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::Mul::kikj(ei, w, e);
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
        Tensor::Mul::ikjk(g.w, dy, x);
        e.zero();
        o1.zero();
        return;
    }

};

}
#endif // LAYER_H
