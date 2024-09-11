#ifndef ATTENTION_HPP
#define ATTENTION_HPP
#include <memory>
#include "tensor.hpp"
#include "ilayer.h"
#include "activate.h"
#include "optimize.h"

namespace RL {

class PositionalEncoder : public iLayer
{
public:
    int inputDim;
    int pos;
    Tensor pe;
public:
    PositionalEncoder(){}
    explicit PositionalEncoder(int inputDim_, bool withGrad_)
        :inputDim(inputDim_)
    {
        o = Tensor(inputDim, 1);
        e = Tensor(inputDim, 1);
    }

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        float d = x.totalSize;
        for (std::size_t i = 0; i < x.totalSize; i++) {
            if (i%2 == 0) {
                pe[i] = std::sin(float(pos)/std::pow(10000, float(i)/d));
            } else {
                pe[i] = std::cos(float(pos)/std::pow(10000, float(i - 1)/d));
            }
            o[i] = x[i] + pe[i];
        }
        return o;
    }
    void gradient(const Tensor& x, const Tensor&) override
    {

    }
    void backward(Tensor &ei) override
    {

    }
    void SGD(float lr) override
    {

    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {

    }
    void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad) override
    {

    }
    void clamp(float c0, float cn) override
    {

    }
    void copyTo(iLayer* layer) override
    {

    }
    void softUpdateTo(iLayer* layer, float alpha) override
    {

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
    bool withMask;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor qk;
    Tensor z;
    Tensor mask;
    ScaledDotProductGrad g;
    ScaledDotProductGrad gv;
    ScaledDotProductGrad gm;
public:
    ScaledDotProduct(){}

    explicit ScaledDotProduct(int inputDim_, int outputDim_, bool withGrad_, bool withMask_=false)
        :inputDim(inputDim_),outputDim(outputDim_),withMask(withMask_)
    {
        type = LAYER_SCALEDDOTPRODUCT;
        wq = Tensor(outputDim, inputDim);
        wk = Tensor(outputDim, inputDim);
        wv = Tensor(outputDim, inputDim);
        Random::uniform(wq, -1, 1);
        Random::uniform(wk, -1, 1);
        Random::uniform(wv, -1, 1);
        q = Tensor(outputDim, 1);
        k = Tensor(outputDim, 1);
        v = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        qk = Tensor(outputDim, outputDim);
        mask = upTriangle(outputDim, outputDim);
        z = Tensor(outputDim, outputDim);
        if (withGrad_) {
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

    static std::shared_ptr<ScaledDotProduct> _(int inputDim, int outputDim, bool withGrad)
    {
        return std::make_shared<ScaledDotProduct>(inputDim, outputDim, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(q, wq, x);
        Tensor::MM::ikkj(k, wk, x);
        Tensor::MM::ikkj(v, wv, x);
        Tensor::MM::ikjk(qk, q, k);
        if (withMask) {
            qk *= mask;
        }
        float d = std::sqrt(outputDim);
        qk /= d;
        z = Softmax::f(qk);
        Tensor::MM::ikkj(o, z, v);
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor w = wq + wk + wv;
        Tensor::MM::kikj(ei, w, e);
        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {
        /*
            q = Wq*x
            k = Wk*x
            v = Wv*x
            z = qk^T/d = Wq*x * (Wk*x)^T/d = Wq*x*(x^T*Wk^T)/d
            o = softmax(z)*v

            do/dz = J(z)*v
            dz/dq = k^T/d
            dz/dk^T = q/d
            dz/dk = q^T/d
            dq/dWq = x^T
            dk/dWk = x^T
            do/dWq = (do/dz)*(dz/dq)*(dq/dWq) = J(z)*(v*k^T/d)*x^T
            do/dWk = (do/dz)*(dz/dk)*(dk/dWk) = J(z)*(v*q^T/d)*x^T
            do/dWv = (do/dv)*(dv/dWv) = sofmax(z)*x^T
        */

        float d = std::sqrt(outputDim);
        /* softmax jacobian */
        Tensor J = Softmax::jacobian(z);
        /* J(z)*(v*k^T/d) */
        Tensor vk(outputDim, outputDim);
        Tensor::MM::ikjk(vk, v, k);
        Tensor jvk(outputDim*outputDim, 1);
        vk.reshape(outputDim*outputDim, 1);
        Tensor::MM::ikkj(jvk, J, vk);
        jvk /= d;
        /* J(z)*(v*q^T/d) */
        Tensor vq(outputDim, outputDim);
        Tensor::MM::ikjk(vq, v, q);
        Tensor jvq(outputDim*outputDim, 1);
        vq.reshape(outputDim*outputDim, 1);
        Tensor::MM::ikkj(jvq, J, vq);
        jvq /= d;

        /* do/dWq, do/dWk, do/dWv */
        Tensor dWq(outputDim, 1);
        Tensor dWk(outputDim, 1);
        Tensor dWv(outputDim, 1);
        jvk.reshape(outputDim, outputDim);
        jvq.reshape(outputDim, outputDim);
        if (withMask) {
            jvk *= mask;
            jvq *= mask;
        }
        Tensor::MM::ikkj(dWq, jvk, e);
        Tensor::MM::ikkj(dWk, jvq, e);
        Tensor::MM::ikkj(dWv, z, e);
        Tensor::MM::ikjk(g.wq, dWq, x);
        Tensor::MM::ikjk(g.wk, dWk, x);
        Tensor::MM::ikjk(g.wv, dWv, x);
        /* zero */
        q.zero();
        k.zero();
        v.zero();
        qk.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(wq, g.wq, lr);
        Optimize::SGD(wk, g.wk, lr);
        Optimize::SGD(wv, g.wv, lr);
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wq, gv.wq, g.wq, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wk, gv.wk, g.wk, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wv, gv.wv, g.wv, lr, rho, decay, clipGrad);
        g.zero();
        return;
    }

     void Adam(float lr, float alpha, float beta,
               float alpha_, float beta_,
               float decay, bool clipGrad) override
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

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<wq.toString()<<std::endl;
         file<<wk.toString()<<std::endl;
         file<<wv.toString()<<std::endl;
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string wqs;
         std::getline(file, wqs);
         wq = Tensor::fromString(wqs);
         std::string wks;
         std::getline(file, wks);
         wk = Tensor::fromString(wks);
         std::string wvs;
         std::getline(file, wvs);
         wv = Tensor::fromString(wvs);
         return;
     }
};

template<int N>
class Attention : public iLayer
{
public:
    class AttentionGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        AttentionGrad(){}
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int unitDim;
    int outputDim;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    ScaledDotProduct dotProduct[N];
    AttentionGrad g;
    AttentionGrad v;
    AttentionGrad m;
public:
    Attention(){}
    explicit Attention(int inputDim_, int unitDim_, bool withGrad)
        :inputDim(inputDim_),unitDim(unitDim_)
    {
        type = LAYER_ATTENTION;
        outputDim = unitDim*N;
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        for (int i = 0; i < N; i++) {
            dotProduct[i] = ScaledDotProduct(inputDim, unitDim, withGrad);
        }
        a = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
            g.w1 = Tensor(outputDim, outputDim);
            v.w1 = Tensor(outputDim, outputDim);
            m.w1 = Tensor(outputDim, outputDim);
            g.w2 = Tensor(outputDim, inputDim);
            v.w2 = Tensor(outputDim, inputDim);
            m.w2 = Tensor(outputDim, inputDim);
            g.b = Tensor(outputDim, 1);
            v.b = Tensor(outputDim, 1);
            m.b = Tensor(outputDim, 1);
        }
    }
    static std::shared_ptr<Attention> _(int inputDim, int outputDim, bool withGrad)
    {
        return std::make_shared<Attention>(inputDim, outputDim, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        for (int i = 0; i < N; i++) {
            Tensor &out = dotProduct[i].forward(x, inference);
            a.embedding({i*unitDim, 0}, out);
        }
        //softmax(a);
        Tensor::MM::ikkj(o, w1, a);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(o[i]);
        }
        return o;
    }

    void backward(Tensor &ei) override
    {
        Tensor::MM::kikj(ei, w2, e);
        Tensor es(e.shape);
        for (int i = 0; i < N; i++) {
            Tensor esi(unitDim, 1);
            dotProduct[i].backward(esi);
            for (int j = 0; j < unitDim; j++) {
                es[unitDim*i + j] += esi[j];
            }
        }
        Tensor::MM::kikj(ei, w1, es);
        return;
    }

    void broadcast() override
    {
        for (int i = 0; i < N; i++) {
            dotProduct[i].e = e.block({unitDim*i, 0}, {unitDim, 1});
        }
        return;
    }

    void gradient(const Tensor& x, const Tensor&y) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dy[i] = Tanh::df(o[i])*e[i];
        }
        Tensor::MM::ikjk(g.w1, dy, a);
        Tensor::MM::ikjk(g.w2, dy, x);
        g.b += dy;
        for (int i = 0; i < N; i++) {
            dotProduct[i].gradient(x, y);
        }
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate) override
    {
        Optimize::SGD(w1, g.w1, learningRate);
        Optimize::SGD(w2, g.w2, learningRate);
        Optimize::SGD(b, g.b, learningRate);
        for (int i = 0; i < N; i++) {
            dotProduct[i].SGD(learningRate);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            dotProduct[i].RMSProp(rho, lr, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w1, v.w1, m.w1, g.w1,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            dotProduct[i].Adam(alpha, beta,
                               alpha_, beta_,
                               lr, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < N; i++) {
             dotProduct[i].clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < N; i++) {
            dotProduct[i].copyTo(&pLayer->dotProduct[i]);
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < N; i++) {
             dotProduct[i].softUpdateTo(&pLayer->dotProduct[i], alpha);
         }
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<w1.toString()<<std::endl;
         file<<w2.toString()<<std::endl;
         /* b */
         file<<b.toString()<<std::endl;
         for (int i = 0; i < N; i++) {
             dotProduct[i].write(file);
         }
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string w1s;
         std::getline(file, w1s);
         w1 = Tensor::fromString(w1s);
         std::string w2s;
         std::getline(file, w2s);
         w2 = Tensor::fromString(w2s);
         std::string bs;
         std::getline(file, bs);
         b = Tensor::fromString(bs);

         for (int i = 0; i < N; i++) {
             dotProduct[i].read(file);
         }
         return;
     }
};
}
#endif // ATTENTION_HPP
