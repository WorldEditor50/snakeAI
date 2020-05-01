#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
namespace ML {
/* activate method */
#define ACTIVATE_SIGMOID 0
#define ACTIVATE_TANH    1
#define ACTIVATE_RELU    2
#define ACTIVATE_LINEAR  3
/* optimize method */
#define OPT_BGD     0
#define OPT_RMSPROP 1
#define OPT_ADAM    2
/* layer type */
#define LOSS_MSE            0
#define LOSS_CROSS_ENTROPY  1
    class Layer {
        public:
            Layer(){}
            ~Layer(){}
            void createLayer(int inputDim, int layerDim, int activateMethod, int lossTye = LOSS_MSE);
            void calculateOutputs(std::vector<double>& x);
            void softmax(std::vector<double>& x);
            void calculateLoss(std::vector<double>& yo, std::vector<double> yt);
            void calculateErrors(std::vector<double>& nextErrors, std::vector<std::vector<double> >& nextWeights);
            void calculateBatchGradient(std::vector<double>& x);
            void calculateSoftmaxGradient(std::vector<double>& x, std::vector<double>& yo, std::vector<double> yt);
            void SGD(std::vector<double>& x, double learningRate);
            void BGD(double learningRate);
            void RMSProp(double rho, double learningRate);
            void Adam(double alpha1, double alpha2, double learningRate);
            std::vector<std::vector<double> > weights;
            std::vector<double> bias;
            std::vector<double> outputs;
            std::vector<double> errors;
            int lossType;
        private:
            double activate(double x);
            double derivativeActivate(double y);
            double sigmoid(double x);
            double dsigmoid(double y);
            double dtanh(double y);
            double relu(double x);
            double drelu(double x);
            double dotProduct(std::vector<double>& x1, std::vector<double>& x2);
            int activateMethod;
            std::vector<std::vector<double> > batchGradientX;
            std::vector<double> batchGradient;
            std::vector<std::vector<double> > sx;
            std::vector<double> sb;
            std::vector<std::vector<double> > vx;
            std::vector<double> vb;
            double alpha1_t;
            double alpha2_t;
            double delta;
            double decay;
    };

    class BPNet {
        public:
            BPNet(){}
            ~BPNet(){}
            void createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int activateMethod);
            void createNetWithSoftmax(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int activateMethod);
            void copyTo(BPNet& dstNet);
            std::vector<double>& getOutput();
            void feedForward(std::vector<double>& xi);
            void backPropagate(std::vector<double>& yo, std::vector<double>& yt);
            void backPropagate(std::vector<double>& loss);
            void calculateBatchGradient(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt);
            void calculateBatchGradient(std::vector<double> &x, std::vector<double> &y);
            void SGD(std::vector<double> &x, std::vector<double> &y, double learningRate);
            void BGD(double learningRate = 0.001);
            void RMSProp(double rho = 0.9, double learningRate = 0.001);
            void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
            void train(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& y,
                    int optimizeMethod,
                    int batchSize,
                    double learningRate,
                    int iterateNum);
            void show();
            void loadParameter(const std::string& fileName);
            void saveParameter(const std::string& fileName);
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
