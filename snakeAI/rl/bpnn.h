#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
namespace RL {
/* activate method */
enum ActiveType {
    SIGMOID = 0,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR
};
/* Optimize method */
enum OptType {
    NONE = 0,
    OPT_SGD,
    OPT_RMSPROP,
    OPT_ADAM
};
/* loss type */
enum LossType {
    MSE = 0,
    CROSS_ENTROPY
};
using Mat = std::vector<std::vector<double> >;
using Vec = std::vector<double>;
class Layer
{
public:
    enum LayerType {
      INPUT = 0,
      HIDDEN,
      OUTPUT
    };
public:
    Layer():inputDim(0), layerDim(0){}
    virtual ~Layer(){}
    explicit Layer(std::size_t inputDim, std::size_t layerDim, LayerType layerType, ActiveType activeType, LossType lossType, bool tarinFlag);
    void feedForward(const Vec &x);
    void loss(const Vec &yo, const Vec &yt);
    void loss(Vec& l);
    void error(const Vec &nextE, const Mat &nextW);
    void gradient(const Vec& x);
    void softmaxGradient(const Vec &x, const Vec &yo, const Vec &yt);
    void SGD(double learningRate);
    void RMSProp(double rho, double learningRate);
    void Adam(double alpha1, double alpha2, double learningRate);
    void RMSPropWithClip(double rho, double learningRate, double threshold);
public:
    /* output */
    Mat W;
    Vec B;
    Vec O;
    Vec E;
    /* paramter */
    std::size_t inputDim;
    std::size_t layerDim;
    ActiveType activeType;
    LossType lossType;
    LayerType layerType;
protected:
    double activate(double x);
    double dActivate(double y);
    void softmax(Vec& x, Vec& y);
    Vec softmax(const Vec &x);
    double max(const Vec& x);
    int argmax(const Vec& x);
    double dotProduct(const Vec &x1, const Vec &x2);
    /* buffer for optimization */
    Mat dW;
    Mat Sw;
    Mat Vw;
    Vec dB;
    Vec Sb;
    Vec Vb;
    double alpha1_t;
    double alpha2_t;
};

class BPNN
{
public:
    BPNN(){}
    virtual ~BPNN(){}
    explicit BPNN(std::size_t inputDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t outputDim, bool trainFlag = false,
            ActiveType activeType = SIGMOID, LossType lossType = MSE);
    void copyTo(BPNN& dstNet);
    void softUpdateTo(BPNN& dstNet, double alpha);
    Vec& output();
    BPNN &feedForward(const Vec &x);
    void backPropagate(const Vec &yo, const Vec &yt);
    void grad(Vec &x, Vec &y, Vec &loss);
    void gradient(const Vec &x, const Vec &y);
    void SGD(double learningRate = 0.001);
    void RMSProp(double rho = 0.9, double learningRate = 0.001);
    void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
    void RMSPropWithClip(double rho = 0.9, double learningRate = 0.001, double threshold = 1);
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
    std::size_t outputIndex;
    std::vector<Layer> layers;
};
}
#endif // BPNN_H
