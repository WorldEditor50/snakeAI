#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "rl_basic.h"

namespace RL {

/* Optimize method */
enum OptType {
    NONE = 0,
    OPT_SGD,
    OPT_RMSPROP,
    OPT_ADAM
};

class LayerParam
{
public:
    /* buffer for optimization */
    Mat dW;
    Mat Sw;
    Mat Vw;
    Vec dB;
    Vec Sb;
    Vec Vb;
    double alpha1_t;
    double alpha2_t;
public:
    LayerParam(){}
    LayerParam(std::size_t inputDim, std::size_t layerDim, bool trainFlag)
    {
        /* buffer for optimization */
        if (trainFlag == true) {
            dW = Mat(layerDim, Vec(inputDim, 0));
            dB = Vec(layerDim, 0);
            Sw = Mat(layerDim, Vec(inputDim, 0));
            Sb = Vec(layerDim, 0);
            Vw = Mat(layerDim, Vec(inputDim, 0));
            Vb = Vec(layerDim, 0);
            alpha1_t = 1;
            alpha2_t = 1;
        }
    }
};

class Layer : public LayerParam
{
public:
    using Activate = std::function<double(double)>;
    using DActivate = std::function<double(double)>;
public:
    Layer(){}
    virtual ~Layer(){}
    explicit Layer(std::size_t inputDim,
                   std::size_t layerDim,
                   Activate activate_,
                   DActivate dActivate_,
                   bool tarinFlag);
    static std::shared_ptr<Layer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    Activate activate_ = Sigmoid::_,
                                    DActivate dActivate_ = Sigmoid::d,
                                    bool tarinFlag = true)
    {
        return std::make_shared<Layer>(inputDim, layerDim, activate_, dActivate_, tarinFlag);
    }
    virtual void feedForward(const Vec &x);
    virtual void gradient(const Vec& x, const Vec&);
    void backpropagate(const Vec &nextE, const Mat &nextW);
    void SGD(double learningRate);
    void RMSProp(double rho, double learningRate);
    void Adam(double alpha1, double alpha2, double learningRate);
public:
    /* output */
    Mat W;
    Vec B;
    Vec O;
    Vec E;
protected:
    Activate activate;
    DActivate dActivate;
};

class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer(){}
    virtual ~SoftmaxLayer(){}
    explicit SoftmaxLayer(std::size_t inputDim, std::size_t layerDim, bool tarinFlag)
        :Layer(inputDim, layerDim, Linear::_, Linear::d, tarinFlag){}
    static std::shared_ptr<SoftmaxLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<SoftmaxLayer>(inputDim, layerDim, tarinFlag);
    }
    virtual void feedForward(const Vec &x) override;
    virtual void gradient(const Vec& x, const Vec& y) override;
};
class MergeLayer : public Layer
{
public:
    MergeLayer(){}
    virtual ~MergeLayer(){}
    explicit MergeLayer(std::size_t inputDim, std::size_t layerDim, bool tarinFlag)
        :Layer(inputDim, layerDim, Linear::_, Linear::d, tarinFlag){}
    static std::shared_ptr<MergeLayer> _(std::size_t inputDim,
                                    std::size_t layerDim,
                                    bool tarinFlag)
    {
        return std::make_shared<MergeLayer>(inputDim, layerDim, tarinFlag);
    }
    virtual void feedForward(const Vec &x) override;
    virtual void gradient(const Vec& x, const Vec& y) override;
};

class BPNN
{
public:
    using LossFunc = std::function<void(Vec&, const Vec&, const Vec&)>;
    using Layers = std::vector<std::shared_ptr<Layer> >;
public:
    BPNN(){}
    virtual ~BPNN(){}
    explicit BPNN(const Layers &layers_):evalTotalError(false),layers(layers_){}
    BPNN(const BPNN &r):evalTotalError(r.evalTotalError),layers(r.layers){}
    BPNN &operator = (const BPNN &r);
    void copyTo(BPNN& dstNet);
    void softUpdateTo(BPNN& dstNet, double alpha);
    Vec& output();
    BPNN &feedForward(const Vec &x);
    double gradient(const Vec &x, const Vec &y, LossFunc loss);
    void SGD(double learningRate = 0.001);
    void RMSProp(double rho = 0.9, double learningRate = 0.001);
    void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
    void optimize(OptType optType = OPT_RMSPROP, double learningRate = 0.001);
    void train(Mat& x,
            Mat& y,
            OptType optType,
            std::size_t batchSize,
            double learningRate,
            std::size_t iterateNum);
    int argmax();
    int argmin();
    void show();
    void load(const std::string& fileName);
    void save(const std::string& fileName);
protected:
    bool evalTotalError;
    Layers layers;
};

}
#endif // BPNN_H
