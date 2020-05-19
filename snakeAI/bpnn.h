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
#define OPT_SGD     0
#define OPT_RMSPROP 1
#define OPT_ADAM    2
/* loss type */
#define LOSS_MSE            0
#define LOSS_CROSS_ENTROPY  1
/* cost type */
#define COST_MSE            0
#define COST_CROSS_ENTROPY  1
    class Layer {
        public:
            Layer(){}
            ~Layer(){}
            void createLayer(int inputDim, int layerDim, int activateType, int lossTye = LOSS_MSE, bool trainFlag = true);
            void feedForward(std::vector<double>& x);
            void feedForward(std::vector<std::vector<double> >& x);
            void softmax(std::vector<double>& x, std::vector<double>& y);
            void calculateLoss(std::vector<double>& yo, std::vector<double> yt);
            void calculateErrors(std::vector<double>& nextE, std::vector<std::vector<double> >& nextW);
            void calculateGradient(std::vector<double>& x);
            void calculateSoftmaxGradient(std::vector<double>& x, std::vector<double>& yo, std::vector<double> yt);
            void SGD(double learningRate);
            void RMSProp(double rho, double learningRate);
            void Adam(double alpha1, double alpha2, double learningRate);
            std::vector<std::vector<double> > W;
            std::vector<double> B;
            std::vector<double> O;
            std::vector<double> E;
            int lossType;
            int layerType;
        private:
            double activate(double x);
            double dActivate(double y);
            double dotProduct(std::vector<double>& x1, std::vector<double>& x2);
            void multiply(std::vector<std::vector<double> >& x1,
                    std::vector<std::vector<double> >& x2,
                    std::vector<std::vector<double> >& y);
            int activateType;
            /* buffer for optimization */
            std::vector<std::vector<double> > dW;
            std::vector<std::vector<double> > Sw;
            std::vector<std::vector<double> > Vw;
            std::vector<double> dB;
            std::vector<double> Sb;
            std::vector<double> Vb;
            double alpha1_t;
            double alpha2_t;
            double delta;
            double decay;
    };

    class BPNet {
        public:
            BPNet(){}
            ~BPNet(){}
            void createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
                    bool trainFlag = true, int activateType = ACTIVATE_SIGMOID, int lossType = LOSS_MSE);
            void copyTo(BPNet& dstNet);
            void softUpdateTo(BPNet& dstNet, double alpha);
            std::vector<double>& getOutput();
            void feedForward(std::vector<double>& xi);
            void backPropagate(std::vector<double>& yo, std::vector<double>& yt);
            void backPropagate(std::vector<double>& loss);
            void calculateGradient(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt);
            void calculateGradient(std::vector<double> &x, std::vector<double> &y);
            void SGD(double learningRate = 0.001);
            void RMSProp(double rho = 0.9, double learningRate = 0.001);
            void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
            void optimize(int optType = OPT_RMSPROP, double learningRate = 0.001);
            void train(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& y,
                    int optType,
                    int batchSize,
                    double learningRate,
                    int iterateNum);
            int maxOutput();
            void show();
            void load(const std::string& fileName);
            void save(const std::string& fileName);
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
