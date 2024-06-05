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

class iLayer
{
public:
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING,
        LAYER_SCALEDDOTPRODUCT
    };
public:
    int type;
    Tensor o;
    Tensor e;
public:
    iLayer(){}
    virtual ~iLayer(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        return o;
    }
    virtual void gradient(const Tensor& x, const Tensor&){}
    virtual void backward(Tensor &ei){}
    virtual void cacheError(const Tensor &e){}
    virtual void SGD(float learningRate){}
    virtual void RMSProp(float rho, float lr, float decay, bool clipGrad){}
    virtual void Adam(float alpha, float beta,
                      float alpha_, float beta_,
                      float lr, float decay, bool clipGrad){}
    virtual void clamp(float c0, float cn){}
    virtual void copyTo(iLayer* layer){}
    virtual void softUpdateTo(iLayer* layer, float alpha){}
};

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

class ScaledDotProduct : public iLayer
{
public:
    class ScaledDotProductGrad
    {
    public:
        Tensor wq;
        Tensor wk;
        Tensor wv;
    public:
        ScaledDotProductGrad(){}
        void zero()
        {
            wq.zero();
            wk.zero();
            wv.zero();
            return;
        }
    };
public:
    int inputDim;
    int outputDim;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor qk;
    Tensor fqk;

    ScaledDotProductGrad g;
    ScaledDotProductGrad gv;
    ScaledDotProductGrad gm;
public:
    ScaledDotProduct(){}

    explicit ScaledDotProduct(int inputDim_, int outputDim_, bool trainFlag_)
        :inputDim(inputDim_),outputDim(outputDim_)
    {
        type = LAYER_SCALEDDOTPRODUCT;
        wq = Tensor(outputDim, inputDim);
        wk = Tensor(outputDim, inputDim);
        wv = Tensor(outputDim, inputDim);
        q = Tensor(outputDim, 1);
        k = Tensor(outputDim, 1);
        v = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        qk = Tensor(outputDim, outputDim);
        fqk = Tensor(outputDim, outputDim);
        if (trainFlag_) {
            g.wq = Tensor(outputDim, inputDim);
            g.wk = Tensor(outputDim, inputDim);
            g.wv = Tensor(outputDim, inputDim);
            gv.wq = Tensor(outputDim, inputDim);
            gv.wk = Tensor(outputDim, inputDim);
            gv.wv = Tensor(outputDim, inputDim);
            gm.wq = Tensor(outputDim, inputDim);
            gm.wk = Tensor(outputDim, inputDim);
            gm.wv = Tensor(outputDim, inputDim);
        }
    }

    static std::shared_ptr<ScaledDotProduct> _(int inputDim, int outputDim, bool tarinFlag)
    {
        return std::make_shared<ScaledDotProduct>(inputDim, outputDim, tarinFlag);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(q, wq, x);
        Tensor::MM::ikkj(k, wk, x);
        Tensor::MM::ikkj(v, wv, x);
        Tensor::MM::ikjk(qk, q, k);
        float d = std::sqrt(outputDim);
        fqk = softmax_(qk);
        Tensor::MM::ikkj(o, fqk, v);
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, wq, e);
        Tensor::MM::kikj(ei, wk, e);
        Tensor::MM::kikj(ei, wv, e);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        /*
            q = Wq*x
            k = Wk*x
            v = Wv*x
            o = f(qk/d)*v

            dq = df(qk/d)*k/d*v
            dk = df(qk/d)*q/d*v
            dv = I
            dWq = dq*e*x
            dWk = dk*e*x
            dWv = f(qk/d)*e*x
        */
        float d = std::sqrt(outputDim);
        Tensor dqk(outputDim, outputDim);
        for (std::size_t i = 0; i < fqk.totalSize; i++) {
            for (std::size_t j = 0; j < qk.totalSize; j++) {
                if (i == j) {
                    dqk[i] += fqk[i]*(1 - fqk[i])*qk[j]/d;
                } else {
                    dqk[i] += -fqk[i]*fqk[j]*qk[j]/d;
                }
            }
        }
        Tensor dq(outputDim, 1);
        Tensor dk(outputDim, 1);
        Tensor::MM::ikkj(dq, dqk, k);
        Tensor::MM::ikkj(dk, dqk, q);
        dq *= v;
        dk *= v;
        dq *= e;
        dk *= e;
        Tensor dv(outputDim, 1);
        Tensor::MM::ikkj(dv, fqk, e);
        Tensor::MM::ikjk(g.wq, dq, x);
        Tensor::MM::ikjk(g.wk, dk, x);
        Tensor::MM::ikjk(g.wv, dv, x);
        q.zero();
        k.zero();
        v.zero();
        qk.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate)
    {
        Optimize::SGD(wq, g.wq, learningRate);
        Optimize::SGD(wk, g.wk, learningRate);
        Optimize::SGD(wv, g.wv, learningRate);
        g.zero();
        return;
    }

    void RMSProp(float rho, float lr, float decay, bool clipGrad)
    {
        Optimize::RMSProp(wq, gv.wq, g.wq, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wk, gv.wk, g.wk, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wv, gv.wv, g.wv, lr, rho, decay, clipGrad);
        g.zero();
        return;
    }

     void Adam(float alpha, float beta,
               float alpha_, float beta_,
               float lr, float decay, bool clipGrad)
    {
        Optimize::Adam(wq, gv.wq, gm.wq, g.wq,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(wk, gv.wk, gm.wk, g.wk,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(wv, gv.wv, gm.wv, g.wv,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(wq, c0, cn);
         Optimize::clamp(wk, c0, cn);
         Optimize::clamp(wv, c0, cn);
         return;
     }

     virtual void copyTo(iLayer* layer) override
     {
         ScaledDotProduct *pLayer = static_cast<ScaledDotProduct*>(layer);
         pLayer->wq = wq;
         pLayer->wk = wk;
         pLayer->wv = wv;
         return;
     }
     virtual void softUpdateTo(iLayer* layer, float alpha) override
     {
         ScaledDotProduct *pLayer = static_cast<ScaledDotProduct*>(layer);
         lerp(pLayer->wq, wq, alpha);
         lerp(pLayer->wk, wk, alpha);
         lerp(pLayer->wv, wv, alpha);
         return;
     }
};

}
#endif // LAYER_H
