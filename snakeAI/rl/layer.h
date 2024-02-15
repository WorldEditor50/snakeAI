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
    virtual void feedForward(const Mat& x){}
    virtual void backward(Mat &){}
    virtual void gradient(const Mat&, const Mat&){}
    virtual void SGD(float ){}
    virtual void RMSProp(float , float , float ){}
    virtual void Adam(float , float ,float , float , float , float ){}
};

class LayerParam : public iLayer
{
public:
    std::size_t inputDim;
    std::size_t layerDim;
    Mat W;
    Mat B;
public:
    LayerParam(){}
    LayerParam(std::size_t inputDim_, std::size_t layerDim_)
        :inputDim(inputDim_), layerDim(layerDim_)
    {
        W = Mat(layerDim, inputDim);
        B = Mat(layerDim, 1);
    }
    void zero()
    {
        W.zero();
        B.zero();
        return;
    }
    void random()
    {
        uniformRand(W, -1, 1);
        uniformRand(B, -1, 1);
        return;
    }

};

class LayerObject : public LayerParam
{
public:
    Mat O;
    Mat E;
    LayerParam s;
    LayerParam v;
    LayerParam d;
public:
    LayerObject(){}
    LayerObject(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerParam(inputDim, layerDim)
    {
        O = Mat(layerDim, 1);
        E = Mat(layerDim, 1);
        if (trainFlag == true) {
            s = LayerParam(inputDim, layerDim);
            v = LayerParam(inputDim, layerDim);
            d = LayerParam(inputDim, layerDim);
        }
        LayerParam::random();
    }

    void backward(Mat &preE) override
    {
        for (std::size_t i = 0; i < W.cols; i++) {
            for (std::size_t j = 0; j < W.rows; j++) {
                preE[i] +=  W(j, i) * E[j];
            }
        }
        return;
    }
    void SGD(float learningRate) override
    {
        Optimizer::SGD(W, d.W, learningRate);
        Optimizer::SGD(B, d.B, learningRate);
        d.zero();
        return;
    }
    void RMSProp(float rho, float learningRate, float decay) override
    {
        Optimizer::RMSProp(W, s.W, d.W, learningRate, rho, decay);
        Optimizer::RMSProp(B, s.B, d.B, learningRate, rho, decay);
        d.zero();
        return;
    }
    void Adam(float alpha, float beta, float alpha_t, float beta_t,float learningRate, float decay)override
    {
        Optimizer::Adam(W, s.W, v.W, d.W,
                        alpha_t, beta_t, learningRate, alpha, beta, decay);
        Optimizer::Adam(B, s.B, v.B, d.B,
                        alpha_t, beta_t, learningRate, alpha, beta, decay);
        d.zero();
        return;
    }
};


template<typename FnActive>
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

    void feedForward(const Mat& x) override
    {
        for (std::size_t i = 0; i < W.rows; i++) {
            for (std::size_t j = 0; j < W.cols; j++) {
                O[i] += W(i, j) * x[j];
            }
            O[i] = FnActive::f(O[i] +  B[i]);
        }
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        for (std::size_t i = 0; i < d.W.rows; i++) {
            float dy = FnActive::d(O[i]) * E[i];
            for (std::size_t j = 0; j < d.W.cols; j++) {
                d.W(i, j) += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
            O[i] = 0;
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
    void feedForward(const RL::Mat &x) override
    {
        float s = 0;
        for (std::size_t i = 0; i < W.rows; i++) {
            for (std::size_t j = 0; j < W.cols; j++) {
                O[i] += W(i, j) * x[j];
            }
            O[i] += B[i];
            s += exp(O[i]);
        }
        for (std::size_t i = 0; i < O.size(); i++) {
            O[i] = exp(O[i]) / s;
        }
        return;
    }

    void gradient(const RL::Mat &x, const Mat &y) override
    {
        for (std::size_t i = 0; i < d.W.rows; i++) {
            float dy = O[i] - y[i];
            for (std::size_t j = 0; j < d.W.cols; j++) {
                d.W(i, j) += dy * x[j];
            }
            d.B[i] += dy;
            O[i] = 0;
        }
        return;
    }

};

class GeluLayer : public LayerObject
{
public:
    Mat O0;
public:
    GeluLayer(){}
    ~GeluLayer(){}
    explicit GeluLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag),O0(layerDim, 1){}

    static std::shared_ptr<GeluLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<GeluLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Mat &x) override
    {
        for (std::size_t i = 0; i < W.rows; i++) {
            for (std::size_t j = 0; j < W.cols; j++) {
                O0[i] += W(i, j) * x[j];
            }
            O0[i] += B[i];
            O[i] = Gelu::f(O0[i]);
        }
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        for (std::size_t i = 0; i < d.W.rows; i++) {
            float dy = Gelu::d(O0[i]) * E[i];
            for (std::size_t j = 0; j < d.W.cols; j++) {
                d.W(i, j) += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
            O[i] = 0;
        }
        return;
    }

};

class SwishLayer : public LayerObject
{
public:
    Mat O0;
public:
    SwishLayer(){}
    ~SwishLayer(){}
    explicit SwishLayer(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
        :LayerObject(inputDim, layerDim, trainFlag),O0(layerDim, 1){}

    static std::shared_ptr<SwishLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SwishLayer>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Mat &x) override
    {
        for (std::size_t i = 0; i < W.rows; i++) {
            for (std::size_t j = 0; j < W.cols; j++) {
                O0[i] += W(i, j) * x[j];
            }
            O0[i] += B[i];
            O[i] = Swish::f(O0[i]);
        }
        return;
    }

    void gradient(const Mat& x, const Mat&) override
    {
        for (std::size_t i = 0; i < d.W.rows; i++) {
            float dy = Swish::d(O0[i]) * E[i];
            for (std::size_t j = 0; j < d.W.cols; j++) {
                d.W(i, j) += dy * x[j];
            }
            d.B[i] += dy;
            E[i] = 0;
            O[i] = 0;
        }
        return;
    }

};

template<typename FnActive>
class DropoutLayer : public Layer<FnActive>
{
public:
    bool trainFlag;
    float p;
    Mat mask;
public:
    DropoutLayer(){}
    ~DropoutLayer(){}
    explicit DropoutLayer(std::size_t inputDim, std::size_t layerDim,
                          bool trainFlag_, float p_)
        :Layer<FnActive>(inputDim, layerDim, trainFlag_),
          trainFlag(trainFlag_), p(p_), mask(layerDim, 1){}

    static std::shared_ptr<DropoutLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag, float p_)
    {
        return std::make_shared<DropoutLayer>(inputDim, layerDim, tarinFlag, p_);
    }
    void feedForward(const RL::Mat &x) override
    {
        Layer<FnActive>::feedForward(x);
        if (trainFlag == true) {
            std::bernoulli_distribution bernoulli(p);
            for (std::size_t i = 0; i < Layer<FnActive>::O.size(); i++) {
                mask[i] = bernoulli(Rand::engine) / (1 - p);
            }
            for (std::size_t i = 0; i < Layer<FnActive>::O.size(); i++) {
                Layer<FnActive>::O[i] *= mask[i];
            }
        }
        return;
    }

    void backward(Mat& preE) override
    {
        if (trainFlag == true) {
            for (std::size_t i = 0; i < Layer<FnActive>::E.size(); i++) {
                Layer<FnActive>::E[i] *= mask[i];
            }
        }
        Layer<FnActive>::backward(preE);
        return;
    }
};


template<typename FnActive>
class LayerNorm : public Layer<FnActive>
{
public:
    float gamma;
    float gamma_;
public:
    LayerNorm(){}
    ~LayerNorm(){}
    explicit LayerNorm(std::size_t inputDim, std::size_t layerDim, bool trainFlag_)
        :Layer<FnActive>(inputDim, layerDim, trainFlag_), gamma(1){}

    static std::shared_ptr<LayerNorm> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<LayerNorm>(inputDim, layerDim, tarinFlag);
    }
    void feedForward(const RL::Mat &x) override
    {
        Mat &O = Layer<FnActive>::O;
        Mat &W = Layer<FnActive>::W;
        Mat &B = Layer<FnActive>::B;
        for (std::size_t i = 0; i < W.rows; i++) {
            for (std::size_t j = 0; j < W.cols; j++) {
                O[i] += W(i, j) * x[j];
            }
        }
        float u = O.mean();
        float sigma = RL::variance(O, u);
        gamma_ = gamma/sqrt(sigma + 1e-9);
        for (std::size_t i = 0; i < O.size(); i++) {
            O[i] = FnActive::f(gamma_*(O[i] - u) + B[i]);
        }

        return;
    }
    void backward(Mat &preE)
    {
        for (std::size_t i = 0; i < Layer<FnActive>::E.size(); i++) {
            Layer<FnActive>::E[i] *= gamma_;
        }
        return Layer<FnActive>::backward(preE);
    }
};

template<typename FnActive>
class BatchNorm : public Layer<FnActive>
{
public:
    bool trainFlag;
    Mat gamma;
    Mat beta;
    std::vector<RL::Mat> x_;
    std::vector<RL::Mat> y;
public:
    explicit BatchNorm(std::size_t inputDim, std::size_t layerDim,
                          bool trainFlag_)
        :Layer<FnActive>(inputDim, layerDim, trainFlag_), trainFlag(trainFlag_)
    {

    }

    void feedForward(const RL::Mat &x) override
    {
        return;
    }
    void feedForward(const std::vector<RL::Mat> &x)
    {
        /* x(batchDim, inputDim)  */
        /* u */
        Mat u(x[0].size(), 1);
        float batchSize = float(x.size());
        for (std::size_t i = 0; i < x[0].size(); i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                u[i] += x[j][i];
            }
            u[i] /= batchSize;
        }
        /* sigma */
        Mat sigma(x[0].size(), 1);
        for (std::size_t i = 0; i < x[0].size(); i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                sigma[i] += (x[j][i] - u[i])*(x[j][i] - u[i]);
            }
        }
        /* x_ = (x - u)/sqrt(sigma + 1e-9) */
        for (std::size_t i = 0; i < x[0].size(); i++) {
            for (std::size_t j = 0; j < x.size(); j++) {
                x_[j][i] = (x[j][i] - u[j])/sqrt(sigma[i] + 1e-9);
            }
        }
        /* y = gamma*x_ + beta */
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[0].size(); j++) {
                y[i][j] = gamma[i]*x_[i][j] + beta[i];
            }
        }
        return;
    }
    void backward(Mat &preE)
    {

        return;
    }
};

}
#endif // LAYER_H
