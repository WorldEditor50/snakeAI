#ifndef CONCAT_HPP
#define CONCAT_HPP
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"

namespace RL {

template<typename TLayer, int N>
class ScaledConcat : public iLayer
{
public:
    class ScaledConcatGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        ScaledConcatGrad(){}
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
    TLayer layers[N];
    ScaledConcatGrad g;
    ScaledConcatGrad v;
    ScaledConcatGrad m;
public:
    ScaledConcat(){}
    explicit ScaledConcat(const TLayer& layer, int inputDim_, int unitDim_, bool withGrad)
        :inputDim(inputDim_),unitDim(unitDim_)
    {
        type = LAYER_SCALEDCONCAT;
        outputDim = unitDim*N;
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        for (int i = 0; i < N; i++) {
            layers[i] = layer;
            layers[i].initParams();
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

    static std::shared_ptr<ScaledConcat> _(const TLayer& layer, int inputDim, int unitDim, bool withGrad)
    {
        return std::make_shared<ScaledConcat>(layer, inputDim, unitDim, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        for (int i = 0; i < N; i++) {
            Tensor &out = layers[i].forward(x, inference);
            a.embedding({i*unitDim, 0}, out);
        }
        softmax(a);
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

        return;
    }

    void broadcast() override
    {
        for (int i = 0; i < N; i++) {
            layers[i].e = e.block({unitDim*i, 0}, {unitDim, 1});
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
            layers[i].gradient(x, y);
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
            layers[i].SGD(learningRate);
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
            layers[i].RMSProp(rho, lr, decay, clipGrad);
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
            layers[i].Adam(alpha, beta, alpha_, beta_, lr, decay, clipGrad);
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
             layers[i].clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         ScaledConcat *pLayer = static_cast<ScaledConcat*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < N; i++) {
            layers[i].copyTo(&pLayer->layers[i]);
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         ScaledConcat *pLayer = static_cast<ScaledConcat*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < N; i++) {
             layers[i].softUpdateTo(&pLayer->layers[i], alpha);
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
             layers[i].write(file);
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
             layers[i].read(file);
         }
         return;
     }
};

class Concat : public iLayer
{
public:
    class ConcatGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        ConcatGrad(){}
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int outputDim;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    std::vector<std::shared_ptr<iLayer> > layers;
    ConcatGrad g;
    ConcatGrad v;
    ConcatGrad m;
public:
    Concat(){}
    template<typename ...TLayer>
    explicit Concat(int inputDim_, bool withGrad, TLayer&&...layer)
        :inputDim(inputDim_),layers({layer...})
    {
        type = LAYER_CONCAT;
        outputDim = 0;
        for (std::size_t i = 0; i < layers.size(); i++) {
            outputDim += layers[i]->o.totalSize;
        }
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
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

    static std::shared_ptr<Concat> _(int inputDim, int unitDim, bool withGrad)
    {
        return std::make_shared<Concat>(inputDim, unitDim, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        int offset = 0;
        for (int i = 0; i < layers.size(); i++) {
            Tensor &out = layers[i]->forward(x, inference);
            a.embedding({offset, 0}, out);
            offset += out.totalSize;
        }
        softmax(a);
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

        return;
    }

    void broadcast() override
    {
        int offset = 0;
        for (int i = 0; i < layers.size(); i++) {
            int unitDim = layers[i]->o.totalSize;
            layers[i]->e = e.block({offset, 0}, {unitDim, 1});
            offset += unitDim;
        }
        return;
    }

    void gradient(const Tensor& x, const Tensor&y) override
    {
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dy[i] = Tanh::df(o[i])*e[i];
        }
        Tensor da(outputDim, 1);
        Tensor J = Softmax::jacobian(a);
        Tensor::MM::ikkj(da, J, dy);
        Tensor::MM::ikjk(g.w1, da, a);
        Tensor::MM::ikjk(g.w2, dy, x);
        g.b += dy;
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->gradient(x, y);
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
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->SGD(learningRate);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(rho, lr, decay, clipGrad);
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
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->Adam(alpha, beta, alpha_, beta_, lr, decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         Concat *pLayer = static_cast<Concat*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < layers.size(); i++) {
            layers[i]->copyTo(pLayer->layers[i].get());
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         Concat *pLayer = static_cast<Concat*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->softUpdateTo(pLayer->layers[i].get(), alpha);
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
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->write(file);
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
         for (int i = 0; i < layers.size(); i++) {
             layers[i]->read(file);
         }
         return;
     }
};
}
#endif // CONCAT_HPP
