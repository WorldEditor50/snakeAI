#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <memory>
#include <iostream>
#include "rl_basic.h"

namespace RL {

class iLayer
{
public:
    virtual ~iLayer(){}
    virtual void feedForward(const Vec& x){}
    virtual void backward(const Vec& nextE, const Mat& nextW){}
    virtual void gradient(const Vec& x, const Vec&){}
    virtual void SGD(double learningRate){}
    virtual void RMSProp(double rho, double learningRate){}
    virtual void Adam(double alpha, double beta, double learningRate){}
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
            O[i] = 0;
            for (std::size_t j = 0; j < W[0].size(); j++) {
                O[i] += W[i][j] * x[j];
            }
            O[i] = ActF::_(O[i] + B[i]);
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

    void backward(const Vec& nextE, const Mat& nextW) override
    {
        if (E.size() != nextW[0].size()) {
            std::cout<<"size is not matching"<<std::endl;;
        }
        for (std::size_t i = 0; i < nextW[0].size(); i++) {
            for (std::size_t j = 0; j < nextW.size(); j++) {
                E[i] +=  nextW[j][i] * nextE[j];
            }
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
            O[i] = RL::dotProduct(W[i], x) + B[i];
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
        std::uniform_real_distribution<double> uniform(0, 1);
        if (trainFlag == true && uniform(Rand::engine) > 0.8) {
            if (p == 0) {
                mask.assign(Layer<ActF>::O.size(), 1);
            } else if (p == 1) {
                mask.assign(Layer<ActF>::O.size(), 0);
            } else {
                std::bernoulli_distribution bernoulli(p);
                for (std::size_t i = 0; i < Layer<ActF>::O.size(); i++) {
                    mask[i] = bernoulli(Rand::engine) / (1 - p);
                }
            }
            for (std::size_t i = 0; i < Layer<ActF>::O.size(); i++) {
                Layer<ActF>::O[i] *= mask[i];
            }
        }
        return;
    }

    void backward(const Vec& nextE, const Mat& nextW) override
    {
        Layer<ActF>::backward(nextE, nextW);
        if (trainFlag == true) {
            for (std::size_t i = 0; i < nextW[0].size(); i++) {
                Layer<ActF>::E[i] *= mask[i];
            }
        }
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
        double u = 0;
        for (std::size_t i = 0; i < O.size(); i++) {
            u += O[i];
        }
        u /= O.size();
        double s = 0;
        for (std::size_t i = 0; i < O.size(); i++) {
            s += (O[i] - u) * (O[i] - u);
        }
        s /= O.size();
        g = 1 / sqrt(s + 1e-9);
        for (std::size_t i = 0; i < O.size(); i++) {
            O[i] = ActF::_(g * (O[i] - u) + B[i]);
        }
        return;
    }
    void backward(const Vec& nextE, const Mat& nextW) override
    {
        Layer<ActF>::backward(nextE, nextW);
        for (std::size_t i = 0; i < Layer<ActF>::E.size(); i++) {
            Layer<ActF>::E[i] *= g;
        }
        return;
    }
};

class BatchNorm
{
public:
    double gamma;
    double beta;
    Vec u;
    Vec s;
    Mat y;
public:
    BatchNorm(){}
    ~BatchNorm(){}
    explicit BatchNorm(std::size_t inputDim, std::size_t layerDim,
                          bool trainFlag_)
        : gamma(1), beta(0), u(layerDim, 0), s(layerDim, 0),
          y(layerDim, Vec(inputDim, 0)){}

    static std::shared_ptr<BatchNorm> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<BatchNorm>(inputDim, layerDim, tarinFlag);
    }
    void forward(const std::vector<RL::Vec> &x)
    {
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[0].size(); j++) {
                u[i] += x[i][j];
            }
            u[i] /= x.size();
        }
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[0].size(); j++) {
                s[i] = (x[i][j] - u[i]) * (x[i][j] - u[i]);
            }
            s[i] /= x.size();
        }
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[0].size(); j++) {
                y[i][j] = gamma * (x[i][j] - u[i]) / sqrt(s[i] + 1e-9) + beta;
            }
        }
        return;
    }
    void backward(const Vec& nextE, const Mat& nextW)
    {

        return;
    }
};

template<typename ActF>
class ResidualLayer : public iLayer
{
public:
    Mat W1;
    Vec B1;
    Mat W2;
    Vec B2;
    Mat W3;
    Vec B3;

    Vec O1;
    Vec E1;
    Vec O2;
    Vec E2;
    Vec O3;
    Vec E3;

    Mat dW1;
    Vec dB1;
    Mat dW2;
    Vec dB2;
    Mat dW3;
    Vec dB3;
public:
    ResidualLayer(){}
    ~ResidualLayer(){}
    explicit ResidualLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
    {
        W1 = Mat(layerDim, Vec(inputDim, 0));
        B1 = Vec(layerDim, 0);
        W2 = Mat(layerDim, Vec(layerDim, 0));
        B2 = Vec(layerDim, 0);
        W3 = Mat(inputDim, Vec(layerDim, 0));
        B3 = Vec(inputDim, 0);

        O1 = Vec(layerDim, 0);
        E1 = Vec(layerDim, 0);
        O2 = Vec(layerDim, 0);
        E2 = Vec(layerDim, 0);
        O3 = Vec(inputDim, 0);
        E3 = Vec(inputDim, 0);

        dW1 = Mat(layerDim, Vec(inputDim, 0));
        dB1 = Vec(layerDim, 0);
        dW2 = Mat(layerDim, Vec(layerDim, 0));
        dB2 = Vec(layerDim, 0);
        dW3 = Mat(inputDim, Vec(layerDim, 0));
        dB3 = Vec(inputDim, 0);
    }

    static std::shared_ptr<SoftmaxLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SoftmaxLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Vec &x) override
    {
        for (std::size_t i = 0; i < W1.size(); i++) {
            double s = RL::dotProduct(W1[i], x);
            O1[i] = ActF::_(s + B1[i]);
        }
        for (std::size_t i = 0; i < W2.size(); i++) {
            double s = RL::dotProduct(W2[i], O1);
            O2[i] = ActF::_(s + B2[i]);
        }
        for (std::size_t i = 0; i < W3.size(); i++) {
            double s = RL::dotProduct(W3[i], O2);
            O3[i] = ActF::_(s + B3[i]) + x[i];
        }
        return;
    }

    void backward(const Vec& nextE, const Mat& nextW) override
    {
        for (std::size_t i = 0; i < nextW[0].size(); i++) {
            for (std::size_t j = 0; j < nextW.size(); j++) {
                E3[i] +=  nextW[j][i] * nextE[j];
            }
        }
        for (std::size_t i = 0; i < W3[0].size(); i++) {
            for (std::size_t j = 0; j < W3.size(); j++) {
                E2[i] +=  W3[j][i] * E3[j];
            }
        }
        for (std::size_t i = 0; i < W2[0].size(); i++) {
            for (std::size_t j = 0; j < W2.size(); j++) {
                E1[i] +=  W2[j][i] * E2[j];
            }
        }
        return;
    }

    void gradient(const Vec& x, const Vec&) override
    {
        for (std::size_t i = 0; i < dW1.size(); i++) {
            double dy = ActF::d(O1[i]) * E1[i];
            for (std::size_t j = 0; j < dW1[0].size(); j++) {
                dW1[i][j] += dy * x[j];
            }
            dB1[i] += dy;
            E1[i] = 0;
        }
        for (std::size_t i = 0; i < dW2.size(); i++) {
            double dy = ActF::d(O2[i]) * E2[i];
            for (std::size_t j = 0; j < dW2[0].size(); j++) {
                dW2[i][j] += dy * O1[j];
            }
            dB2[i] += dy;
            E2[i] = 0;
        }
        for (std::size_t i = 0; i < dW3.size(); i++) {
            double dy = ActF::d(O3[i]) * E3[i] + 1;
            for (std::size_t j = 0; j < dW3[0].size(); j++) {
                dW3[i][j] += dy * O2[j];
            }
            dB3[i] += dy;
            E3[i] = 0;
        }
        return;
    }

    void SGD(double learningRate)
    {
        Optimizer::SGD(dW1, W1, learningRate);
        Optimizer::SGD(dW2, W2, learningRate);
        Optimizer::SGD(dW3, W3, learningRate);

        Optimizer::SGD(dB1, B1, learningRate);
        Optimizer::SGD(dB2, B2, learningRate);
        Optimizer::SGD(dB3, B3, learningRate);
        return;
    }

    void copyTo(ResidualLayer &dst)
    {
        for (std::size_t i = 0; i < W1.size(); i++) {
            for (std::size_t j = 0; j < W1[0].size(); j++) {
                dst.W1[i][j] =  W1[j][i];
            }
            dst.B1[i] = B1[i];
        }
        for (std::size_t i = 0; i < W2.size(); i++) {
            for (std::size_t j = 0; j < W2[0].size(); j++) {
                dst.W2[i][j] =  W2[j][i];
            }
            dst.B2[i] = B2[i];
        }
        for (std::size_t i = 0; i < W3.size(); i++) {
            for (std::size_t j = 0; j < W3[0].size(); j++) {
                dst.W3[i][j] =  W3[j][i];
            }
            dst.B3[i] = B3[i];
        }
    }

    void softUpdateTo(ResidualLayer &dst, double rho)
    {
        for (std::size_t i = 0; i < W1.size(); i++) {
            RL::EMA(dst.W1[i], W1[i], rho);
        }
        RL::EMA(dst.B1, B1, rho);
        for (std::size_t i = 0; i < W2.size(); i++) {
            RL::EMA(dst.W2[i], W2[i], rho);
        }
        RL::EMA(dst.B2, B2, rho);
        for (std::size_t i = 0; i < W3.size(); i++) {
            RL::EMA(dst.W3[i], W3[i], rho);
        }
        RL::EMA(dst.B3, B3, rho);
    }
};

}
#endif // LAYER_H
