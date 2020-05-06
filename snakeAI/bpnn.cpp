#include "bpnn.h"
namespace ML {

    double Layer::dotProduct(std::vector<double>& x1, std::vector<double>& x2)
    {
        double sum = 0;
        for (int i = 0; i < x1.size(); i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }
            
    void Layer::multiply(std::vector<std::vector<double> >& x1,
                    std::vector<std::vector<double> >& x2,
                    std::vector<std::vector<double> >& y)
    {
        if (x1[0].size() != x2.size()) {
            return;
        }
        if (y[0].size() != x1[0].size() || y.size() != x2.size()) {
            return;
        }
        /* (m, p) x (p, n) = (m, n) */
        int m = x1[0].size();
        int p = x2[0].size();
        int n = x2.size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    y[i][j] += x1[i][k] * x2[k][j];
                }
            }
        }
        return;
    }

    void Layer::softmax(std::vector<double>& x, std::vector<double>& y)
    {
        double s = 0;
        double maxValue = x[0];
        for (int i = 0; i < x.size(); i++) {
            if (x[i] > maxValue) {
                maxValue = x[i];
            }
        }
        for (int i = 0; i < x.size(); i++) {
            s += exp(x[i] - maxValue);
        }
        for (int i = 0; i < x.size(); i++) {
            y[i] = exp(x[i] - maxValue) / s;
        }
        return;
    }

    double Layer::activate(double x)
    {
        double y = 0;
        switch (activateType) {
            case ACTIVATE_SIGMOID:
                y = exp(x) / (exp(x) + 1);
                break;
            case ACTIVATE_RELU:
                y = x > 0 ? x : 0;
                break;
            case ACTIVATE_TANH:
                y = tanh(x);
                break;
            case ACTIVATE_LINEAR:
                y = x;
                break;
            default:
                y = exp(x) / (exp(x) + 1);
                break;
        }
        return y;
    }

    double Layer::dActivate(double y)
    {
        double dy = 0;
        switch (activateType) {
            case ACTIVATE_SIGMOID:
                dy = y * (1 - y);
                break;
            case ACTIVATE_RELU:
                dy = y  > 0 ? 1 : 0;
                break;
            case ACTIVATE_TANH:
                dy = 1 - y * y;
                break;
            case ACTIVATE_LINEAR:
                dy = 1;
                break;
            default:
                dy = y * (1 - y);
                break;
        }
        return dy;
    }

    void Layer::createLayer(int inputDim, int layerDim, int activateType, int lossType)
    {
        if (layerDim < 1 || inputDim < 1) {
            return;
        }
        this->lossType = lossType;
        this->activateType = activateType;
        W.resize(layerDim);
        B.resize(layerDim);
        O.resize(layerDim, 0);
        E.resize(layerDim, 0);
        /* buffer for optimization */
        dW.resize(layerDim);
        dB.resize(layerDim);
        Sw.resize(layerDim);
        Sb.resize(layerDim, 0);
        Vw.resize(layerDim);
        Vb.resize(layerDim, 0);
        this->alpha1_t = 1;
        this->alpha2_t = 1;
        this->delta = pow(10, -8);
        this->decay = 0;
        /* init */
        for (int i = 0; i < W.size(); i++) {
            W[i].resize(inputDim);
            dW[i].resize(inputDim);
            Sw[i].resize(inputDim, 0);
            Vw[i].resize(inputDim, 0);
        }
        /* init */
        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[0].size(); j++) {
                W[i][j] = double(rand() % 10000 - rand() % 10000) / 10000;
            }
            B[i] = double(rand() % 10000 - rand() % 10000) / 10000;
        }
        return;
    }

    void Layer::feedForward(std::vector<double>& x)
    {
        if (x.size() != W[0].size()) {
            std::cout<<"not same size"<<std::endl;
            return;
        }
        double y = 0;
        for (int i = 0; i < W.size(); i++) {
            y = dotProduct(W[i], x) + B[i];
            O[i] = activate(y);
        }
        if (lossType == LOSS_CROSS_ENTROPY) {
            softmax(O, O);
        }
        return;
    }

    void Layer::calculateErrors(std::vector<double>& nextE, std::vector<std::vector<double> >& nextW)
    {
        if (E.size() != nextW[0].size()) {
            std::cout<<"size is not matching"<<std::endl;;
        }
        for (int i = 0; i < nextW[0].size(); i++) {
            for (int j = 0; j < nextW.size(); j++) {
                E[i] += nextE[j] * nextW[j][i];   
            }
        }
        return;
    }

    void Layer::calculateLoss(std::vector<double>& yo, std::vector<double> yt)
    {
        for (int i = 0; i < yo.size(); i++) {
            if (lossType == LOSS_CROSS_ENTROPY) {
                E[i] = -yt[i] * log(yo[i]);
            } else if (lossType == LOSS_MSE){
                E[i] = yo[i] - yt[i];
            }
        }
        return;
    }

    void Layer::calculateGradient(std::vector<double>& x)
    {
        for (int i = 0; i < dW.size(); i++) {
            double dy = dActivate(O[i]);
            for (int j = 0; j < dW[0].size(); j++) {
                dW[i][j] += E[i] * dy * x[j]; 
            }
            dB[i] += E[i] * dy; 
            E[i] = 0;
        }
        return;
    }

    void Layer::calculateSoftmaxGradient(std::vector<double>& x, std::vector<double>& yo, std::vector<double> yt)
    {
        for (int i = 0; i < dW.size(); i++) {
            double dOutput = yo[i] - yt[i];
            for (int j = 0; j < dW[0].size(); j++) {
                dW[i][j] += dOutput * x[j];
            }
            dB[i] += dOutput;
            E[i] = 0;
        }
        return;
    }

    void Layer::SGD(double learningRate)
    {
        /*
         * e = (Activate(wx + b) - T)^2/2
         * de/dw = (Activate(wx +b) - T)*DActivate(wx + b) * x
         * de/db = (Activate(wx +b) - T)*DActivate(wx + b)
         * */
        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[0].size(); j++) {
                W[i][j] += decay * W[i][j] - learningRate * dW[i][j];
                dW[i][j] = 0;
            }
            B[i] += decay * B[i] - learningRate * dB[i];
            dB[i] = 0;
        }
        return;
    }

    void Layer::RMSProp(double rho, double learningRate)
    {
        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[0].size(); j++) {
                Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
                W[i][j] += decay * W[i][j] - learningRate * dW[i][j] / (sqrt(Sw[i][j]) + delta);
                dW[i][j] = 0;
            }
            Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
            B[i] += decay * B[i] - learningRate * dB[i] / (sqrt(Sb[i]) + delta);
            dB[i] = 0;
        }
        return;
    }

    void Layer::Adam(double alpha1, double alpha2, double learningRate)
    {
        double v;
        double s;
        for (int i = 0; i < W.size(); i++) {
            alpha1_t *= alpha1;
            alpha2_t *= alpha2;
            for (int j = 0; j < W[0].size(); j++) {
                /* momentum */
                Vw[i][j] = alpha1 * Vw[i][j] + (1 - alpha1) * dW[i][j];
                /* delcay factor */
                Sw[i][j] = alpha2 * Sw[i][j] + (1 - alpha2) * dW[i][j] * dW[i][j];
                v = Vw[i][j] / (1 - alpha1_t);
                s = Sw[i][j] / (1 - alpha2_t);
                W[i][j] += decay * W[i][j] - learningRate * v / (sqrt(s) + delta);
                dW[i][j] = 0;
            }
            Vb[i] = alpha1 * Vb[i] + (1 - alpha1) * dB[i];
            Sb[i] = alpha2 * Sb[i] + (1 - alpha2) * dB[i] * dB[i];
            v = Vb[i] / (1 - alpha1_t);
            s = Sb[i] / (1 - alpha2_t);
            B[i] += decay * B[i] - learningRate * v / (sqrt(s) + delta);
            dB[i] = 0;
        }
        return;
    }

    void BPNet::createNet(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim, int activateType, int lossType)
    {
        Layer inputLayer;
        inputLayer.createLayer(inputDim, hiddenDim, activateType);
        layers.push_back(inputLayer);
        for (int i = 1; i < hiddenLayerNum; i++) {
            Layer hiddenLayer;
            hiddenLayer.createLayer(hiddenDim, hiddenDim, activateType);
            layers.push_back(hiddenLayer);
        }
        if (lossType == LOSS_MSE) {
            Layer outputLayer;
            outputLayer.createLayer(hiddenDim, outputDim, activateType);
            layers.push_back(outputLayer);
        }
        if (lossType == LOSS_CROSS_ENTROPY) {
            Layer softmaxLayer;
            softmaxLayer.createLayer(hiddenDim, outputDim, ACTIVATE_LINEAR, LOSS_CROSS_ENTROPY);
            layers.push_back(softmaxLayer);
        }
        this->outputIndex = layers.size() - 1;
        return;
    }

    void BPNet::copyTo(BPNet& dstNet)
    {
        if (layers.size() != dstNet.layers.size()) {
            return;
        }
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].W.size(); j++) {
                for (int k = 0; k < layers[i].W[j].size(); k++) {
                    dstNet.layers[i].W[j][k] = layers[i].W[j][k];
                }
                dstNet.layers[i].B[j] = layers[i].B[j];
            }
        }
        return;
    }

    void BPNet::softUpdateTo(BPNet &dstNet, double alpha)
    {
        if (layers.size() != dstNet.layers.size()) {
            return;
        }
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].W.size(); j++) {
                for (int k = 0; k < layers[i].W[j].size(); k++) {
                    dstNet.layers[i].W[j][k] = (1 - alpha) * dstNet.layers[i].W[j][k] + alpha * layers[i].W[j][k];
                }
                dstNet.layers[i].B[j] = (1 - alpha) * dstNet.layers[i].B[j] + alpha * layers[i].B[j];
            }
        }
        return;
    }

    void BPNet::feedForward(std::vector<double>& xi)
    {
        layers[0].feedForward(xi);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].feedForward(layers[i - 1].O);
        }
        return;
    }

    std::vector<double>& BPNet::getOutput()
    {
        std::vector<double>& outputs = layers[outputIndex].O;
        return outputs;
    }

    void BPNet::backPropagate(std::vector<double>& yo, std::vector<double>& yt)
    {
        /* calculate loss */
        layers[outputIndex].calculateLoss(yo, yt);
        /* error backpropagate */
        for (int i = outputIndex - 1; i >= 0; i--) {
            layers[i].calculateErrors(layers[i + 1].E, layers[i + 1].W);
        }
        return;
    }

    void BPNet::backPropagate(std::vector<double> &loss)
    {
        if (loss.size() != layers[outputIndex].E.size()) {
            return;
        }
        layers[outputIndex].E = loss;
        /* error backpropagate */
        for (int i = outputIndex - 1; i >= 0; i--) {
            layers[i].calculateErrors(layers[i + 1].E, layers[i + 1].W);
        }
        return;
    }

    void BPNet::calculateGradient(std::vector<double> &x, std::vector<double> &yo, std::vector<double> &yt)
    {
        backPropagate(yo, yt);
        /* calculate  gradient */
        layers[0].calculateGradient(x);
        for (int j = 1; j < layers.size(); j++) {
            if (layers[j].lossType == LOSS_CROSS_ENTROPY) {
                layers[j].calculateSoftmaxGradient(layers[j - 1].O, yo, yt);
            } else {
                layers[j].calculateGradient(layers[j - 1].O);
            }
        }
        return;
    }

    void BPNet::calculateGradient(std::vector<double> &x, std::vector<double> &y)
    {
        feedForward(x);
        backPropagate(layers[outputIndex].O, y);
        /* calculate  gradient */
        layers[0].calculateGradient(x);
        for (int j = 1; j < layers.size(); j++) {
            if (layers[j].lossType == LOSS_CROSS_ENTROPY) {
                layers[j].calculateSoftmaxGradient(layers[j - 1].O, layers[outputIndex].O, y);
            } else {
                layers[j].calculateGradient(layers[j - 1].O);
            }
        }
        return;
    }

    void BPNet::SGD(double learningRate)
    {
        /* gradient descent */
        for (int i = 0; i < layers.size(); i++) {
            layers[i].SGD(learningRate);
        }
        return;
    }

    void BPNet::RMSProp(double rho, double learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].RMSProp(rho, learningRate);
        }
        return;
    }

    void BPNet::Adam(double alpha1, double alpha2, double learningRate)
    {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].Adam(alpha1, alpha2, learningRate);
        }
        return;
    }

    void BPNet::optimize(int optType, double learningRate)
    {
        switch (optType) {
            case OPT_SGD:
                SGD(learningRate);
                break;
            case OPT_RMSPROP:
                RMSProp(0.9, learningRate);
                break;
            case OPT_ADAM:
                Adam(0.9, 0.99, learningRate);
                break;
            default:
                RMSProp(0.9, learningRate);
                break;
        }
        return;
    }

    void BPNet::train(std::vector<std::vector<double> >& x,
            std::vector<std::vector<double> >& y,
            int optType,
            int batchSize,
            double learningRate,
            int iterateNum)
    {
        if (x.empty() || y.empty()) {
            std::cout<<"x or y is empty"<<std::endl;
            return;
        }
        if (x.size() != y.size()) {
            std::cout<<"x != y"<<std::endl;
            return;
        }
        if (x[0].size() != layers[0].W[0].size()) {
            std::cout<<"x != w"<<std::endl;
            return;
        }
        if (y[0].size() != layers[outputIndex].O.size()) {
            std::cout<<"y != output"<<std::endl;
            return;
        }
        int len = x.size();
        for (int i = 0; i < iterateNum; i++) {
            for (int j = 0; j < batchSize; j++) {
                int k = rand() % len;
                calculateGradient(x[k], y[k]);
            }
            optimize(optType, learningRate);
        }
        return;
    }

    int BPNet::maxOutput()
    {
        int index = 0;
        double maxValue = layers[outputIndex].O[0];
        for (int i = 0; i < layers[outputIndex].O.size(); i++) {
            if (maxValue < layers[outputIndex].O[i]) {
                maxValue = layers[outputIndex].O[i];
                index = i;
            }
        }
        return index;
    }

    void BPNet::show()
    {
        for (int i = 0; i < layers[outputIndex].O.size(); i++) {
            std::cout<<layers[outputIndex].O[i]<<" ";
        }
        std::cout<<std::endl;;
        return;
    }

    void BPNet::load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].W.size(); j++) {
                for (int k = 0; k < layers[i].W[j].size(); k++) {
                    file >> layers[i].W[j][k];
                }
                file >> layers[i].B[j];
            }
        }
        return;
    }

    void BPNet::save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName);
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i].W.size(); j++) {
                for (int k = 0; k < layers[i].W[j].size(); k++) {
                    file << layers[i].W[j][k];
                }
                file << layers[i].B[j];
                file << std::endl;
            }
        }
        return;
    }
}
