#ifndef LAYER_H
#define LAYER_H
#include <memory>
#include <iostream>
#include <fstream>
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
    bool bias;
    Tensor w;
    Tensor b;
    FcGrad g;
    FcGrad v;
    FcGrad m;
public:
    iFcLayer(){}
    iFcLayer(std::size_t inputDim_, std::size_t outputDim_, bool bias_, bool withGrad)
        :inputDim(inputDim_), outputDim(outputDim_), bias(bias_)
    {
        type = iLayer::LAYER_FC;
        w = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
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
    virtual void initParams() override
    {
        Random::uniform(w, -1, 1);
        Random::uniform(b, -1, 1);
        return;
    }
    virtual Tensor& forward(const Tensor& x, bool inference=false) override
    {
        Tensor::MM::ikkj(o, w, x);
        if (bias) {
            o += b;
        }
        return o;
    }

    virtual void backward(const Tensor& x, Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w, e);
        /* gradient*/
        Tensor::MM::ikjk(g.w, e, x);
        if (bias) {
            g.b += e;
        }
        e.zero();
        o.zero();
        return;
    }
    virtual void SGD(float lr) override
    {
        Optimize::SGD(w, g.w, lr, true);
        if (bias) {
            Optimize::SGD(b, g.b, lr, true);
        }
        g.zero();
        return;
    }

    virtual void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w, v.w, g.w, lr, rho, decay, clipGrad);
        if (bias) {
            Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    virtual void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad) override
    {
        Optimize::Adam(w, v.w, m.w, g.w,
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
    virtual void clamp(float c0, float cn) override
    {
        Optimize::clamp(w, c0, cn);
        if (bias) {
            Optimize::clamp(b, c0, cn);
        }
        return;
    }
    virtual void copyTo(iLayer* layer) override
    {
        iFcLayer *pLayer = static_cast<iFcLayer*>(layer);
        pLayer->w = w;
        if (bias) {
            pLayer->b = b;
        }
        return;
    }
    virtual void softUpdateTo(iLayer* layer, float alpha) override
    {
        iFcLayer *pLayer = static_cast<iFcLayer*>(layer);
        lerp(pLayer->w, w, alpha);
        if (bias) {
            lerp(pLayer->b, b, alpha);
        }
        return;
    }

    virtual void write(std::ofstream &file) override
    {
        /* w */
        file<<w.toString()<<std::endl;
        /* b */
        file<<b.toString()<<std::endl;
        return;
    }

    virtual void read(std::ifstream &file) override
    {
        /* w */
        std::string ws;
        std::getline(file, ws);
        w = Tensor::fromString(ws);
        /* b */
        std::string bs;
        std::getline(file, bs);
        b = Tensor::fromString(bs);
        return;
    }
};

template<typename Fn>
class Layer : public iFcLayer
{
public:
    using Type = Layer<Fn>;
public:
    Layer(){}
    virtual ~Layer(){}
    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool bias,
                                    bool withGrad)
    {
        return std::make_shared<Layer>(inputDim, outputDim, bias, withGrad);
    }

    Layer(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_){}

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        Tensor::MM::ikkj(o, w, x);
        if (bias) {
            o += b;
        }
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Fn::f(o[i]);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        for (std::size_t i = 0; i < e.totalSize; i++) {
            e[i] = Fn::df(o[i]) * e[i];
        }
        Tensor::MM::kikj(ei, w, e);

        Tensor::MM::ikjk(g.w, e, x);
        if (bias) {
            g.b += e;
        }
        e.zero();
        o.zero();
        return;
    }

};

template<>
class Layer<Softmax> : public iFcLayer
{
public:
    Layer() {}
    ~Layer(){}
    explicit Layer(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_)
    {

    }

    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool bias,
                                    bool withGrad)
    {
        return std::make_shared<Layer>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o, w, x);
        if (bias) {
            o += b;
        }
        return softmax(o);
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        Tensor dy(outputDim, 1);
        Softmax::jacobian_transpose_mul(o, e, dy);
        Tensor::MM::kikj(ei, w, dy);
        /* gradient */
        Tensor::MM::ikjk(g.w, dy, x);
        if (bias) {
            g.b += dy;
        }
        e.zero();
        o.zero();
        return;
    }

};

template<>
class Layer<Gelu> : public iFcLayer
{
public:
    Tensor op;
public:
    Layer(){}
    ~Layer(){}
    explicit Layer(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_),op(outputDim, 1){}

    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool bias,
                                    bool withGrad)
    {
        return std::make_shared<Layer>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        op.zero();
        Tensor::MM::ikkj(op, w, x);
        if (bias) {
            op += b;
        }
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Gelu::f(op[i]);
        }
        return o;
    }

    void backward(const RL::Tensor &x, Tensor &ei) override
    {
        for (std::size_t i = 0; i < e.totalSize; i++) {
            e[i] = Gelu::df(op[i]) * e[i];
        }
        Tensor::MM::kikj(ei, w, e);
        /* gradient */
        Tensor::MM::ikjk(g.w, e, x);
        if (bias) {
            g.b += e;
        }
        e.zero();
        o.zero();
        return;
    }

};

template<>
class Layer<Swish> : public iFcLayer
{
public:
    Tensor op;
public:
    Layer(){}
    ~Layer(){}
    explicit Layer(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_),op(outputDim, 1){}

    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool bias,
                                    bool withGrad)
    {
        return std::make_shared<Layer>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        op.zero();
        Tensor::MM::ikkj(op, w, x);
        if (bias) {
            op += b;
        }
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Swish::f(op[i]);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        for (std::size_t i = 0; i < e.totalSize; i++) {
            e[i] = Swish::df(op[i]) * e[i];
        }
        Tensor::MM::kikj(ei, w, e);
        /* gradient */
        Tensor::MM::ikjk(g.w, e, x);
        if (bias) {
            g.b += e;
        }
        e.zero();
        o.zero();
        return;
    }

};

template<>
class Layer<Mish> : public iFcLayer
{
public:
    Tensor op;
public:
    Layer(){}
    ~Layer(){}
    explicit Layer(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_),op(outputDim, 1){}

    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t outputDim,
                                    bool bias,
                                    bool withGrad)
    {
        return std::make_shared<Layer>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        op.zero();
        Tensor::MM::ikkj(op, w, x);
        if (bias) {
            op += b;
        }
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Mish::f(op[i]);
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        for (std::size_t i = 0; i < e.totalSize; i++) {
            e[i] = Mish::df(op[i]) * e[i];
        }
        Tensor::MM::kikj(ei, w, e);
        /* gradient */
        Tensor::MM::ikjk(g.w, e, x);
        if (bias) {
            g.b += e;
        }
        e.zero();
        o.zero();
        return;
    }


};

template<typename Fn>
class Dropout : public Layer<Fn>
{
public:
    bool withGrad;
    float p;
    Tensor mask;
public:
    Dropout(){}
    ~Dropout(){}
    explicit Dropout(std::size_t inputDim, std::size_t outputDim, bool bias_,
                          bool withGrad_, float p_)
        :Layer<Fn>(inputDim, outputDim, bias_, withGrad_),
          withGrad(withGrad_), p(p_), mask(outputDim, 1){}

    static std::shared_ptr<Dropout> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool bias,
                                      bool withGrad, float p)
    {
        return std::make_shared<Dropout>(inputDim, outputDim, bias, withGrad, p);
    }
    Tensor& forward(const Tensor &x, bool inference=false) override
    {
        Layer<Fn>::forward(x);
        if (withGrad == true) {
            std::bernoulli_distribution bernoulli(p);
            for (std::size_t i = 0; i < Layer<Fn>::o.size(); i++) {
                mask[i] = bernoulli(Random::engine) / (1 - p);
            }
            Layer<Fn>::o *= mask;
        }
        return Layer<Fn>::o;
    }

    void backward(const Tensor &x, Tensor& ei) override
    {
        if (withGrad == true) {
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
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool bias,
                                        bool withGrad)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        u = op.mean();
        float sigma = op.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        if (bias) {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f((op[i] - u)*gamma + b[i]);
            }
        } else {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f((op[i] - u)*gamma);
            }
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* Forward: o = Fn(gamma·(op - u) + b)
           backward: dy = gamma · (I - 1/n·1·1^T) · (Fn'(o) · e)
        */
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < e.totalSize; i++) {
            dy[i] = Fn::df(o[i]) * e[i];          // activation derivative => dz
        }
        float u = dy.mean();
        for (std::size_t i = 0; i < e.totalSize; i++) {
            e[i] = gamma * (dy[i] - u);     // LN centering => dL/dop
        }
        Tensor::MM::kikj(ei, w, dy);

        /* gradient() independently computes activation derivative */
        if (bias) {
            g.b += dy;                     // dL/db = Fn'(o)·e (NOT LN-centered)
        }
        /* Recompute LN-centered dy for weight gradient */
        Tensor::MM::ikjk(g.w, e, x);
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
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_), gamma(1)
    {
        x_ = Tensor(inputDim, 1);
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool bias,
                                        bool withGrad)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, bias, withGrad);
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
        if (bias) {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f(op[i] + b[i]);
            }
        } else {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f(op[i]);
            }
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* Forward is: x -> zscore -> w·x_ + b -> Fn::f */
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < e.totalSize; i++) {
            dy[i] = Fn::df(o[i]) * e[i];           // dy = Fn'(o)·e
        }
        /* Propagate through w: dL/dx_ = w^T · dy */
        Tensor dL(outputDim, 1);
        Tensor::MM::kikj(dL, w, dy);
        /* Propagate through zscore: dL/dx = gamma · (I - 1/n·1·1^T) · dq */
        float u = dL.mean();
        for (std::size_t i = 0; i < ei.totalSize; i++) {
            ei[i] = gamma * (dL[i] - u);
        }

        /* gradient() uses x_ (z-scored input) but recomputes dy independently */
        Tensor::MM::ikjk(g.w, dy, x_);
        if (bias) {
            g.b += dy;
        }
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
    explicit LayerNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_), gamma(1)
    {
        o1 = Tensor(outputDim, 1);
        o2 = Tensor(outputDim, 1);
    }

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                        std::size_t outputDim,
                                        bool bias,
                                        bool withGrad)
    {
        return std::make_shared<LayerNorm>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o1, w, x);
        if (bias) {
            for (std::size_t i = 0; i < o.size(); i++) {
                o2[i] = Fn::f(o1[i] + b[i]);
            }
        } else {
            for (std::size_t i = 0; i < o.size(); i++) {
                o2[i] = Fn::f(o1[i]);
            }
        }
        u = o2.mean();
        float sigma = o2.variance(u);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < o.size(); i++) {
            o[i] = (o2[i] - u)*gamma;
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /* Forward: o1 = w·x + b → o2 = Fn(o1) → o = gamma·(o2 - u) */
        Tensor dy(outputDim, 1);
        float u = e.mean();
        for (std::size_t i = 0; i < e.totalSize; i++) {
            dy[i] = gamma * (e[i] - u) * Fn::df(o2[i]);           // activation derivative
        }
        Tensor::MM::kikj(ei, w, dy);                  // linear backward

        Tensor::MM::ikjk(g.w, dy, x);
        if (bias) {
            for (std::size_t i = 0; i < e.totalSize; i++) {
                g.b[i] += Fn::df(o2[i]) * dy[i];      // dL/db = Fn'(o2)·e (pre-LN)
            }
        }

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
    explicit RMSNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_), gamma(1)
    {
        op = Tensor(outputDim, 1);
    }

    static std::shared_ptr<RMSNorm> _(std::size_t inputDim,
                                      std::size_t outputDim,
                                      bool bias,
                                      bool withGrad)
    {
        return std::make_shared<RMSNorm>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(op, w, x);
        float sigma = op.variance(0);
        gamma = 1.0/std::sqrt(sigma + 1e-9);
        if (bias) {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f(gamma*op[i] + b[i]);
            }
        } else {
            for (std::size_t i = 0; i < o.size(); i++) {
                o[i] = Fn::f(gamma*op[i]);
            }
        }
        return o;
    }

    void backward(const RL::Tensor &x, Tensor &ei) override
    {
        /* Forward: o = Fn(gamma·op + b) */
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < e.totalSize; i++) {
            dy[i] = gamma * Fn::df(o[i]) * e[i];     // dy = gamma · Fn'(o) · e
        }
        Tensor::MM::kikj(ei, w, dy);

        if (bias) {
            for (std::size_t i = 0; i < e.totalSize; i++) {
                g.b[i] += Fn::df(o[i]) * e[i];      // dL/db = Fn'(o)·e (not gamma-scaled)
            }
        }
        /* Recompute gamma-scaled dy for weight gradient */
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
    float r;
    Tensor o1;
    Tensor o2;
public:
    TanhNorm(){}
    explicit TanhNorm(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
        :iFcLayer(inputDim, outputDim, bias_, withGrad_)
    {
        o1 = Tensor(outputDim, 1);
        o2 = Tensor(outputDim, 1);
        r = 1.0 - 1.0/float(outputDim);
    }

    static std::shared_ptr<TanhNorm> _(std::size_t inputDim,
                                       std::size_t outputDim,
                                       bool bias,
                                       bool withGrad)
    {
        return std::make_shared<TanhNorm>(inputDim, outputDim, bias, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(o1, w, x);
        o1 *= r;
        o2 = RL::tanh(o1);
        if (bias) {
            for (std::size_t i = 0; i < o.totalSize; i++) {
                o[i] = Fn::f(o2[i] + b[i]);
            }
        } else {
            for (std::size_t i = 0; i < o.totalSize; i++) {
                o[i] = Fn::f(o2[i]);
            }
        }
        return o;
    }

    void backward(const Tensor &x, Tensor &ei) override
    {
        /*
            Forward chain:
                o1 = w · x
                o2 = tanh(o1 * r)
                o  = Fn::f(o2 + b)

            dy = dL/do1 = r · (1 - tanh²(o2)) · Fn'(o) · e

            Save original e (dL/do) for bias gradient computation.
        */
        Tensor dL(outputDim, 1);
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < e.totalSize; i++) {
            float d1 = Fn::df(o[i]) * e[i];       // Fn'(o) · e
            float d2 = o2[i];
            dL[i] = r * (1 - d2 * d2) * d1;         // tanh'(o2) · r · d1 = dL/do1
            dy[i] = d1;
        }
        Tensor::MM::kikj(ei, w, dL);

        /*
         * e[i] = dL/do1 = r * tanh'(o2[i]) * Fn'(o[i]) * dL/do
         * We use e for weight gradient: g.w += e · x^T = dL/do1 · x^T
         * For bias gradient, we need Fn'(o[i]) * dL/do from saved e0:
         *   g.b[i] += Fn::df(o[i]) * e0[i]
         * where e0 = dL/do (original gradient before backward modified it).
         */
        Tensor::MM::ikjk(g.w, dL, x);
        if (bias) {
            g.b += dy;
        }
        e.zero();
        o1.zero();
        return;
    }

};

}
#endif // LAYER_H
