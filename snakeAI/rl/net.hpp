#ifndef NET_HPP
#define NET_HPP
#if 0
#include <memory>
#include <cmath>
#include "tensor.hpp"

struct Sigmoid {
    inline static float f(float x) {return 1/(1 + std::exp(-x));}
    inline static float df(float y) {return y*(1 - y);}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = 1/(1 + std::exp(-x.val[i]));
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i]*(1 - y.val[i]);
        }
        return;
    }

};

struct Tanh {
    inline static float f(float x) {return std::tanh(x);}
    inline static float df(float y) {return 1 - y*y;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = std::tanh(x.val[i]);
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = 1 - y.val[i]*y.val[i];
        }
        return;
    }
};

struct Relu {
    inline static float f(float x) {return x > 0 ? x : 0;}
    inline static float df(float y) {return y > 0 ? 1 : 0;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0;
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0;
        }
        return;
    }
};

struct LeakyRelu {
    inline static float f(float x) {return x > 0 ? x : 0.01*x;}
    inline static float df(float y) {return y > 0 ? 1 : 0.01;}
    inline static void f(Tensor &x)
    {
        for (std::size_t i = 0; i < x.totalSize; i++) {
            x.val[i] = x.val[i] > 0 ? x.val[i] : 0.01*x.val[i];
        }
        return;
    }

    inline static void df(Tensor &y)
    {
        for (std::size_t i = 0; i < y.totalSize; i++) {
            y.val[i] = y.val[i] > 0 ? 1 : 0.01;
        }
        return;
    }
};

struct Linear {
    inline static float f(float x) {return x;}
    inline static float df(float) {return 1;}
    inline static void f(Tensor &x){}

    inline static void df(Tensor &y)
    {
        y.fill(1);
        return;
    }
};

struct Swish {
    static constexpr float beta = 1.0;//1.702;
    inline static float f(float x) {return x*Sigmoid::f(beta*x);}
    inline static float d(float x)
    {
        float s = Sigmoid::f(beta*x);
        return s + x*s*(1 - s);
    }
};

struct Gelu {
    static constexpr float c1 = 0.79788456080287;/* sqrt(2/pi) */
    static constexpr float c2 = 0.044715;
    inline static float f(float x)
    {
        return 0.5*x*(1 + tanh(c1*(x + c2*x*x*x)));
    }
    inline static float df(float x)
    {
        float t = std::tanh(c1*(x + c2*x*x*x));
        return 0.5*(1 + t + x*(c1*(1 + 3*c2*x*x)*(1 - t*t)));
    }
};

struct Loss
{
    static Tensor MSE(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yp.shape);
        loss = yp - yt;
        loss *= 2;
        return loss;
    }

    static Tensor CrossEntropy(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yp.shape);
        for (std::size_t i = 0; i < yp.totalSize; i++) {
            loss[i] = -yt[i] * std::log(yp[i]);
        }
        return loss;
    }
    static Tensor BCE(const Tensor& yp, const Tensor& yt)
    {
        Tensor loss(yt.shape);
        for (std::size_t i = 0; i < yp.totalSize; i++) {
            loss[i] = -(yt[i] * std::log(yp[i]) + (1 - yt[i]) * std::log(1 - yp[i]));
        }
        return loss;
    }
};

class SGD
{
public:
    float decay;
public:
    SGD():decay(0){}
    explicit SGD(const std::vector<int> &):decay(0){}
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        for (std::size_t i = 0; i < w.totalSize; i++) {
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i];
        }
        dw.zero();
        return;
    }
};

class RMSProp
{
public:
    float decay;
    float rho;
    Tensor v;
public:
    RMSProp():decay(0.0f),rho(0.9f){}
    explicit RMSProp(const std::vector<int> &shape)
        :decay(0.0f),rho(0.9f)
    {
        v = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        for (std::size_t i = 0; i < w.totalSize; i++) {
            v.val[i] = rho*v.val[i] + (1 - rho) * dw.val[i]*dw.val[i];
            w.val[i] = (1 - decay)*w.val[i] - learningRate*dw.val[i]/(std::sqrt(v.val[i]) + 1e-9);
        }
        dw.zero();
        return;
    }
};

class Adam
{
public:
    float decay;
    float alpha;
    float beta;
    float alpha_;
    float beta_;
    Tensor v;
    Tensor m;
public:
    Adam():decay(0),alpha(0.9f),beta(0.99f){}
    explicit Adam(const std::vector<int> &shape)
        :decay(0),alpha(0.9f),beta(0.99f),alpha_(1),beta_(1)
    {
        v = Tensor(shape);
        m = Tensor(shape);
    }
    inline void operator()(Tensor& w, Tensor& dw, float learningRate)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        for (std::size_t i = 0; i < w.totalSize; i++) {
            m[i] = alpha*m[i] + (1 - alpha)*dw[i];
            v[i] = beta*v[i] + (1 - beta)*dw[i]*dw[i];
            float m_ = m[i]/(1 - alpha_);
            float v_ = v[i]/(1 - beta_);
            w[i] = (1 - decay)*w[i] - learningRate*m_/(std::sqrt(v_) + 1e-9);
        }
        dw.zero();
        return;
    }
};

class Layer
{
public:
    int type;
    bool bias;
    Tensor w;
    Tensor b;
    Tensor o;
public:
    Layer(){}
    explicit Layer(std::vector<int> &shape, bool bias_)
        :w(shape)
    {

    }
    virtual Tensor& forward(const Tensor &x)
    {
        Tensor::Mul::ikkj(o, w, b);
        return o;
    }
    virtual void copyTo(Layer& dst)
    {

    }
    virtual void softUpdateTo(Layer &dst, float rho)
    {

        return;
    }
};

class Softmax
{
public:

};

class LayerNorm
{
public:

};

class Dropout
{
public:

};


class Net
{
public:
    std::vector<std::shared_ptr<Layer> > layers;
};

struct Grad {
    Tensor w;
    Tensor b;
    Tensor e;
};

template<typename Optim>
class Optimizer
{
private:
    Net &net;
public:
    explicit Optimizer(Net &net_, float lr=1e-3, float decay=0)
    {

    }
    void backward()
    {

    }
    void update()
    {

    }
};
#endif
#endif // NET_HPP
