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
            void CreateLayer(int inputDim, int layerDim, int activateType, int lossTye,  int tarinFlag);
            void FeedForward(std::vector<float>& x);
            void Loss(std::vector<float>& yo, std::vector<float> yt);
            void Error(std::vector<float>& nextE, std::vector<std::vector<float> >& nextW);
            void Gradient(std::vector<float>& x);
            void Gradient(std::vector<float>& x, float threshold);
            void SoftmaxGradient(std::vector<float>& x, std::vector<float>& yo, std::vector<float>& yt);
            void SGD(float learningRate);
            void RMSProp(float rho, float learningRate);
            void Adam(float alpha1, float alpha2, float learningRate);
            void RMSPropWithClip(float rho, float learningRate, float threshold);
            std::vector<std::vector<float> > W;
            std::vector<float> B;
            std::vector<float> O;
            std::vector<float> E;
            int inputDim;
            int layerDim;
            int lossType;
            int layerType;
        private:
            float Activate(float x);
            float dActivate(float y);
            void softmax(std::vector<float>& x, std::vector<float>& y);
            float dotProduct(std::vector<float>& x1, std::vector<float>& x2);
            int activateType;
            /* buffer for optimization */
            std::vector<std::vector<float> > dW;
            std::vector<std::vector<float> > Sw;
            std::vector<std::vector<float> > Vw;
            std::vector<float> dB;
            std::vector<float> Sb;
            std::vector<float> Vb;
            float alpha1_t;
            float alpha2_t;
            float delta;
            float decay;
    };

    class BPNet {
        public:
            BPNet(){}
            ~BPNet(){}
            void CreateNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int trainFlag = 0,
                    int activateType = ACTIVATE_SIGMOID, int lossType = LOSS_MSE);
            void CopyTo(BPNet& dstNet);
            void SoftUpdateTo(BPNet& dstNet, float alpha);
            std::vector<float>& GetOutput();
            int FeedForward(std::vector<float>& x);
            void BackPropagate(std::vector<float>& yo, std::vector<float>& yt);
            void Gradient(std::vector<float> &x, std::vector<float> &y);
            void SGD(float learningRate = 0.001f);
            void RMSProp(float rho = 0.9, float learningRate = 0.001f);
            void Adam(float alpha1 = 0.9, float alpha2 = 0.99, float learningRate = 0.001f);
            void RMSPropWithClip(float rho = 0.9, float learningRate = 0.001f, float threshold = 1);
            void Optimize(int optType = OPT_RMSPROP, float learningRate = 0.001f);
            void Train(std::vector<std::vector<float> >& x,
                    std::vector<std::vector<float> >& y,
                    int optType,
                    int batchSize,
                    float learningRate,
                    int iterateNum);
            int Argmax();
            void Show();
            void Load(const std::string& fileName);
            void Save(const std::string& fileName);
            int lossType;
            int outputIndex;
            std::vector<Layer> layers;
    };
}
#endif // BPNN_H
