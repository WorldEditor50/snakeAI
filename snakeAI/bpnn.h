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
/* Optimize method */
#define OPT_NONE    0
#define OPT_SGD     1
#define OPT_RMSPROP 2
#define OPT_ADAM    3
/* loss type */
#define LOSS_MSE            0
#define LOSS_ENTROPY        1
#define LOSS_CROSS_ENTROPY  2
#define LOSS_KL             3
    class Layer {
        public:
            Layer():inputDim(0), layerDim(0){}
            ~Layer(){}
            void createLayer(int inputDim, int layerDim, int activateType, int lossTye,  int tarinFlag);
            void feedForward(std::vector<double>& x);
            void loss(std::vector<double>& yo, std::vector<double>& yt);
            void loss(std::vector<double>& l);
            void error(std::vector<double>& nextE, std::vector<std::vector<double> >& nextW);
            void gradient(std::vector<double>& x);
            void gradient(std::vector<double>& x, double threshold);
            void softmaxGradient(std::vector<double>& x, std::vector<double>& yo, std::vector<double>& yt);
            void SGD(double learningRate);
            void RMSProp(double rho, double learningRate);
            void Adam(double alpha1, double alpha2, double learningRate);
            void RMSPropWithClip(double rho, double learningRate, double threshold);
            std::vector<std::vector<double> > W;
            std::vector<double> B;
            std::vector<double> O;
            std::vector<double> E;
            int inputDim;
            int layerDim;
            int lossType;
            int layerType;
        private:
            double activate(double x);
            double dActivate(double y);
            void softmax(std::vector<double>& x, std::vector<double>& y);
            double dotProduct(std::vector<double>& x1, std::vector<double>& x2);
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
            void createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int trainFlag = 0,
                    int activateType = ACTIVATE_SIGMOID, int lossType = LOSS_MSE);
            void copyTo(BPNet& dstNet);
            void softUpdateTo(BPNet& dstNet, double alpha);
            std::vector<double>& getOutput();
            int feedForward(std::vector<double>& x);
            void backPropagate(std::vector<double>& yo, std::vector<double>& yt);
            void grad(std::vector<double> &x, std::vector<double> &loss);
            void gradient(std::vector<double> &x, std::vector<double> &y);
            void SGD(double learningRate = 0.001);
            void RMSProp(double rho = 0.9, double learningRate = 0.001);
            void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
            void RMSPropWithClip(double rho = 0.9, double learningRate = 0.001, double threshold = 1);
            void optimize(int optType = OPT_RMSPROP, double learningRate = 0.001);
            void train(std::vector<std::vector<double> >& x,
                    std::vector<std::vector<double> >& y,
                    int optType,
                    int batchSize,
                    double learningRate,
                    int iterateNum);
            int argmax();
            void show();
            void load(const std::string& fileName);
            void save(const std::string& fileName);
            int lossType;
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
