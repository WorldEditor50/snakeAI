#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <memory>
#include <iostream>
#include "util.h"
#include "optimizer.h"
#include "activate.h"
#include "loss.h"

namespace RL {

class iLayer
{
public:
    virtual ~iLayer(){}
    virtual void feedForward(const Vec& x){}
    virtual void backward(Vec &preE){}
    virtual void gradient(const Vec& x, const Vec&){}
    virtual void SGD(double learningRate){}
    virtual void RMSProp(double rho, double learningRate){}
    virtual void Adam(double alpha, double beta,double alpha_t, double beta_t, double learningRate){}
};

class LayerParam : public iLayer
{
public:
    Mat W;
    Vec B;
public:
    LayerParam(){}
    LayerParam(std::size_t inputDim, std::size_t layerDim)
    {
        W = Mat(layerDim, Vec(inputDim, 0));
        B = Vec(layerDim, 0);
    }
    void zero()
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = 0;
            }
            B[i] = 0;
        }
        return;
    }
    void random()
    {
        std::uniform_real_distribution<double> distributionReal(-1, 1);
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = distributionReal(Rand::engine);
            }
            B[i] = distributionReal(Rand::engine);
        }
        return;
    }

};

class LayerObject : public LayerParam
{
public:
    Vec O;
    Vec E;
    LayerParam s;
    LayerParam v;
    LayerParam d;
public:
    LayerObject(){}
    LayerObject(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerParam(inputDim, layerDim)
    {
        O = Vec(layerDim);
        E = Vec(layerDim);
        if (trainFlag == true) {
            s = LayerParam(inputDim, layerDim);
            v = LayerParam(inputDim, layerDim);
            d = LayerParam(inputDim, layerDim);
        }
        LayerParam::random();
    }

    void backward(Vec &preE) override
    {
        for (std::size_t i = 0; i < W[0].size(); i++) {
            for (std::size_t j = 0; j < W.size(); j++) {
                preE[i] +=  W[j][i] * E[j];
            }
        }
        return;
    }
    void SGD(double learningRate) override
    {
        Optimizer::SGD(W, d.W, learningRate);
        Optimizer::SGD(B, d.B, learningRate);
        d.zero();
        return;
    }
    void RMSProp(double rho, double learningRate) override
    {
        Optimizer::RMSProp(W, s.W, d.W, learningRate, rho);
        Optimizer::RMSProp(B, s.B, d.B, learningRate, rho);
        d.zero();
        return;
    }
    void Adam(double alpha, double beta, double alpha_t, double beta_t,double learningRate)override
    {
        Optimizer::Adam(W, s.W, v.W, d.W,
                        alpha_t, beta_t, learningRate, alpha, beta);
        Optimizer::Adam(B, s.B, v.B, d.B,
                        alpha_t, beta_t, learningRate, alpha, beta);
        d.zero();
        return;
    }
};


template<typename ActF>
class Layer : public LayerObject
{
public:
    Layer(){}
    virtual ~Layer(){}
    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag = true)
    {
        return std::make_shared<Layer>(inputDim, layerDim, tarinFlag);
    }

    Layer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag){}

    void feedForward(const Vec& x) override
    {
        if (x.size() != W[0].size()) {
            std::cout<<"x = "<<x.size()<<std::endl;
            std::cout<<"w = "<<W[0].size()<<std::endl;
            return;
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            O[i] = RL::dot(W[i], x) + B[i];
            O[i] = ActF::_(O[i]);
        }
        return;
    }

    void gradient(const Vec& x, const Vec&) override
    {
        for (std::size_t i = 0; i < d.W.size(); i++) {
            double dy = ActF::d(O[i]) * E[i];
            for (std::size_t j = 0; j < d.W[0].size(); j++) {
                d.W[i][j] += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
        }
        return;
    }
};

class SoftmaxLayer : public LayerObject
{
public:
    SoftmaxLayer(){}
    ~SoftmaxLayer(){}
    explicit SoftmaxLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag){}

    static std::shared_ptr<SoftmaxLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SoftmaxLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Vec &x) override
    {
        double s = 0;
        for (std::size_t i = 0; i < W.size(); i++) {
            O[i] = RL::dot(W[i], x) + B[i];
            s += exp(O[i]);
        }
        for (std::size_t i = 0; i < O.size(); i++) {
            O[i] = exp(O[i]) / s;
        }
        return;
    }

    void gradient(const RL::Vec &x, const Vec &y) override
    {
        for (std::size_t i = 0; i < d.W.size(); i++) {
            double dy = O[i] - y[i];
            for (std::size_t j = 0; j < d.W[0].size(); j++) {
                d.W[i][j] += dy * x[j];
            }
            d.B[i] += dy;
        }
        return;
    }

};

class GeluLayer : public LayerObject
{
public:
    Vec O0;
public:
    GeluLayer(){}
    ~GeluLayer(){}
    explicit GeluLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag),O0(layerDim, 0){}

    static std::shared_ptr<GeluLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<GeluLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Vec &x) override
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            O0[i] = RL::dot(W[i], x) + B[i];
            O[i] = Gelu::_(O0[i]);
        }
        return;
    }

    void gradient(const Vec& x, const Vec&) override
    {
        for (std::size_t i = 0; i < d.W.size(); i++) {
            double dy = Gelu::d(O0[i]) * E[i];
            for (std::size_t j = 0; j < d.W[0].size(); j++) {
                d.W[i][j] += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
        }
        return;
    }

};

class SwishLayer : public LayerObject
{
public:
    Vec O0;
public:
    SwishLayer(){}
    ~SwishLayer(){}
    explicit SwishLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag),O0(layerDim, 0){}

    static std::shared_ptr<SwishLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SwishLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Vec &x) override
    {
        for (std::size_t i = 0; i < W.size(); i++) {
            O0[i] = RL::dot(W[i], x) + B[i];
            O[i] = Swish::_(O0[i]);
        }
        return;
    }

    void gradient(const Vec& x, const Vec&) override
    {
        for (std::size_t i = 0; i < d.W.size(); i++) {
            double dy = Swish::d(O0[i]) * E[i];
            for (std::size_t j = 0; j < d.W[0].size(); j++) {
                d.W[i][j] += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
        }
        return;
    }

};

template<typename ActF>
class DropoutLayer : public Layer<ActF>
{
public:
    bool trainFlag;
    double p;
    Vec mask;
public:
    DropoutLayer(){}
    ~DropoutLayer(){}
    explicit DropoutLayer(std::size_t inputDim, std::size_t layerDim,
                          bool trainFlag_, double p_)
        :Layer<ActF>(inputDim, layerDim, trainFlag_),
          trainFlag(trainFlag_), p(p_), mask(layerDim, 1){}

    static std::shared_ptr<DropoutLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag, double p_)
    {
        return std::make_shared<DropoutLayer>(inputDim, layerDim, tarinFlag, p_);
    }
    void feedForward(const RL::Vec &x) override
    {
        Layer<ActF>::feedForward(x);
        if (trainFlag == true) {
            std::bernoulli_distribution bernoulli(p);
            for (std::size_t i = 0; i < Layer<ActF>::O.size(); i++) {
                mask[i] = bernoulli(Rand::engine) / (1 - p);
            }
            for (std::size_t i = 0; i < Layer<ActF>::O.size(); i++) {
                Layer<ActF>::O[i] *= mask[i];
            }
        }
        return;
    }

    void backward(Vec& preE) override
    {
        if (trainFlag == true) {
            for (std::size_t i = 0; i < Layer<ActF>::E.size(); i++) {
                Layer<ActF>::E[i] *= mask[i];
            }
        }
        Layer<ActF>::backward(preE);
        return;
    }
};


template<typename ActF>
class LayerNorm : public Layer<ActF>
{
public:
    double g;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t layerDim,
                          bool trainFlag_)
        :Layer<ActF>(inputDim, layerDim, trainFlag_), g(1){}

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Vec &x) override
    {
        Vec &O = Layer<ActF>::O;
        Mat &W = Layer<ActF>::W;
        Vec &B = Layer<ActF>::B;
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                O[i] += W[i][j] * x[j];
            }
        }
        double u = RL::mean(O);
        double s = RL::variance(O, u);
        g = 1 / sqrt(s + 1e-9);
        for (std::size_t i = 0; i < O.size(); i++) {
            O[i] = ActF::_(g * (O[i] - u) + B[i]);
        }
        return;
    }
    void backward(Vec &preE)
    {
        for (std::size_t i = 0; i < Layer<ActF>::E.size(); i++) {
            Layer<ActF>::E[i] *= g;
        }
        return Layer<ActF>::backward(preE);
    }
};

}
#endif // LAYER_H
