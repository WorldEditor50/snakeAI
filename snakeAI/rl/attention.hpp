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
        Random::uniform(wq, -1, 1);
        Random::uniform(wk, -1, 1);
        Random::uniform(wv, -1, 1);
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
        qk /= d;
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

            dq = df(qk/d)*qk/d*k*v
            dk = df(qk/d)*qk/d*q*v
            dv = I
            dWq = dq*e*x
            dWk = dk*e*x
            dWv = f(qk/d)*e*x
        */

        /* softmax jacobi */
        Tensor dqk(outputDim, outputDim);
        for (std::size_t i = 0; i < outputDim; i++) {
            for (std::size_t j = 0; j < outputDim; j++) {
                if (i == j) {
                    dqk(i, j) = fqk(i, j)*(1 - fqk(i, j));
                } else {
                    dqk(i, j) = -fqk(i, j)*fqk(i, j);
                }
            }
        }
        dqk *= qk;
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

    void SGD(float learningRate) override
    {
        Optimize::SGD(wq, g.wq, learningRate);
        Optimize::SGD(wk, g.wk, learningRate);
        Optimize::SGD(wv, g.wv, learningRate);
        g.zero();
        return;
    }

    void RMSProp(float rho, float lr, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wq, gv.wq, g.wq, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wk, gv.wk, g.wk, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wv, gv.wv, g.wv, lr, rho, decay, clipGrad);
        g.zero();
        return;
    }

     void Adam(float alpha, float beta,
               float alpha_, float beta_,
               float lr, float decay, bool clipGrad) override
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
        Random::uniform(w, -1, 1);
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
    static std::shared_ptr<Attention> _(int inputDim, int outputDim, bool tarinFlag)
    {
        return std::make_shared<Attention>(inputDim, outputDim, tarinFlag);
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

    void RMSProp(float rho, float lr, float decay, bool clipGrad)
    {
        Optimize::RMSProp(w, v.w, g.w, lr, rho, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            dotProduct[i].RMSProp(rho, lr, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void Adam(float alpha, float beta,
               float alpha_, float beta_,
               float lr, float decay, bool clipGrad)
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
