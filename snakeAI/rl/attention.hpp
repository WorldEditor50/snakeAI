#ifndef ATTENTION_HPP
#define ATTENTION_HPP
#include <memory>
#include "tensor.hpp"
#include "ilayer.h"
#include "activate.h"
#include "optimize.h"

namespace RL {

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
    Tensor z;

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
        z = Tensor(outputDim, outputDim);
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
        float d = std::sqrt(outputDim);
        qk /= d;
        z = Softmax::f(qk);
        Tensor::MM::ikkj(o, z, v);
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
            z = qk^T/d = Wq*x * (Wk*x)^T/d = Wq*x*(x^T*Wk^T)/d
            o = softmax(z)*v

            do/dz = dSoftmax(z)*v
            dz/dq = k^T/d
            dz/dk^T = q/d
            dz/dk = q^T/d
            dq/dWq = x^T
            dk/dWk = x^T
            do/dWq = (do/dz)*(dz/dq)*(dq/dWq) = dSoftmax(z)*(v*k^T/d)*x^T
            do/dWk = (do/dz)*(dz/dk)*(dk/dWk) = dSoftmax(z)*(v*q^T/d)*x^T
            do/dWv = (do/dv)*(dv/dWv) = sofmax(z)*x^T
        */

        float d = std::sqrt(outputDim);
        /* softmax jacobian */
        Tensor J = Softmax::jacobian(z);
        Tensor vk(outputDim, outputDim);
        Tensor::MM::ikjk(vk, v, k);
        Tensor jvk(outputDim*outputDim, 1);
        vk.reshape(outputDim*outputDim, 1);
        Tensor::MM::ikkj(jvk, J, vk);

        Tensor vq(outputDim, outputDim);
        Tensor::MM::ikjk(vq, v, q);
        Tensor jvq(outputDim*outputDim, 1);
        vq.reshape(outputDim*outputDim, 1);
        Tensor::MM::ikkj(jvq, J, vq);

        jvq /= d;
        jvk /= d;
        Tensor dWq(outputDim, 1);
        Tensor dWk(outputDim, 1);
        Tensor dWv(outputDim, 1);
        jvk.reshape(outputDim, outputDim);
        jvq.reshape(outputDim, outputDim);
        Tensor::MM::ikkj(dWq, jvk, e);
        Tensor::MM::ikkj(dWk, jvq, e);
        Tensor::MM::ikkj(dWv, z, e);
        Tensor::MM::ikjk(g.wq, dWq, x);
        Tensor::MM::ikjk(g.wk, dWk, x);
        Tensor::MM::ikjk(g.wv, dWv, x);

        q.zero();
        k.zero();
        v.zero();
        qk.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate) override
    {
        Optimize::SGD(wq, g.wq, learningRate);
        Optimize::SGD(wk, g.wk, learningRate);
        Optimize::SGD(wv, g.wv, learningRate);
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
};

template<int N>
class Attention : public iLayer
{
public:
    class AttentionGrad
    {
    public:
        Tensor w;
    public:
        AttentionGrad(){}
        void zero()
        {
            w.zero();
        }
    };
public:
    int inputDim;
    int outputDim;
    Tensor w;
    Tensor a;
    ScaledDotProduct dotProduct[N];
    AttentionGrad g;
    AttentionGrad v;
    AttentionGrad m;
public:
    Attention(){}
    explicit Attention(int inputDim_, int outputDim_, bool trainFlag)
        :inputDim(inputDim_),outputDim(outputDim_)
    {
        w = Tensor(outputDim, outputDim);
        for (int i = 0; i < N; i++) {
            dotProduct[i] = ScaledDotProduct(inputDim, outputDim/N, trainFlag);
        }
        a = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (trainFlag) {
            g.w = Tensor(outputDim, outputDim);
            v.w = Tensor(outputDim, outputDim);
            m.w = Tensor(outputDim, outputDim);
        }
    }
    static std::shared_ptr<Attention> _(int inputDim, int outputDim, bool withGrad)
    {
        return std::make_shared<Attention>(inputDim, outputDim, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        int unit = outputDim/N;
        for (int i = 0; i < N; i++) {
            Tensor &out = dotProduct[i].forward(x, inference);
            a.embedding({i*unit, 0}, out);
        }
        Tensor::MM::ikkj(o, w, a);
        return o;
    }

    void backward(Tensor &ei) override
    {

        return;
    }

    void gradient(const Tensor& x, const Tensor&y) override
    {
        g.w = e*a;
        int unit = outputDim/N;
        for (int i = 0; i < N; i++) {
            dotProduct[i].e = e.block({unit*i, 0}, {unit, 1});
            dotProduct[i].gradient(x, y);
        }
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate)
    {
        Optimize::SGD(w, g.w, learningRate);
        for (int i = 0; i < N; i++) {
            dotProduct[i].SGD(learningRate);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w, v.w, g.w, lr, rho, decay, clipGrad);
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
        Optimize::Adam(w, v.w, m.w, g.w,
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
         Optimize::clamp(w, c0, cn);
         for (int i = 0; i < N; i++) {
             dotProduct[i].clamp(c0, cn);
         }
         return;
     }

     virtual void copyTo(iLayer* layer) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         pLayer->w = w;
         for (int i = 0; i < N; i++) {
            dotProduct[i].copyTo(&pLayer->dotProduct[i]);
         }
         return;
     }
     virtual void softUpdateTo(iLayer* layer, float alpha) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         lerp(pLayer->w, w, alpha);
         for (int i = 0; i < N; i++) {
             dotProduct[i].softUpdateTo(&pLayer->dotProduct[i], alpha);
         }
         return;
     }

};
}
#endif // ATTENTION_HPP
