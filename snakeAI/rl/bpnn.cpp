#include "bpnn.h"

RL::Layer::Layer(std::size_t inputDim,
                 std::size_t layerDim,
                 Activate activate_,
                 DActivate dActivate_,
                 bool trainFlag)
    :LayerParam(inputDim, layerDim, trainFlag)
{
    activate = activate_;
    dActivate = dActivate_;
    W = Mat(layerDim, Vec(inputDim, 0));
    B = Vec(layerDim);
    O = Vec(layerDim);
    E = Vec(layerDim);
    /* init */
    std::uniform_real_distribution<double> distributionReal(-1, 1);
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            W[i][j] = distributionReal(Rand::engine);
        }
        B[i] = distributionReal(Rand::engine);
    }
    return;
}

void RL::Layer::feedForward(const Vec& x)
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
        O[i] = activate(O[i] + B[i]);
    }
    return;
}

void RL::Layer::gradient(const Vec& x, const Vec&)
{
    for (std::size_t i = 0; i < dW.size(); i++) {
        double dy = dActivate(O[i]) * E[i];
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            dW[i][j] += dy * x[j];
        }
        dB[i] += dy;
        E[i] = 0;
    }
    return;
}

void RL::Layer::backpropagate(const Vec& nextE, const Mat& nextW)
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

void RL::Layer::SGD(double learningRate)
{
    /*
     * e = (Activate(wx + b) - T)^2/2
     * de/dw = (Activate(wx +b) - T)*DActivate(wx + b) * x
     * de/db = (Activate(wx +b) - T)*DActivate(wx + b)
     * */
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            W[i][j] -= learningRate * dW[i][j];
            dW[i][j] = 0;
        }
        B[i] -= learningRate * dB[i];
        dB[i] = 0;
    }
    return;
}

void RL::Layer::RMSProp(double rho, double learningRate)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
            W[i][j] -= learningRate * dW[i][j] / (sqrt(Sw[i][j]) + 1e-9);
            dW[i][j] = 0;
        }
        Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
        B[i] -= learningRate * dB[i] / (sqrt(Sb[i]) + 1e-9);
        dB[i] = 0;
    }
    return;
}

void RL::Layer::Adam(double alpha1, double alpha2, double learningRate)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            /* momentum */
            Vw[i][j] = alpha1 * Vw[i][j] + (1 - alpha1) * dW[i][j];
            /* delcay factor */
            Sw[i][j] = alpha2 * Sw[i][j] + (1 - alpha2) * dW[i][j] * dW[i][j];
            double v = Vw[i][j] / (1 - alpha1_t);
            double s = Sw[i][j] / (1 - alpha2_t);
            W[i][j] -= learningRate * v / (sqrt(s) + 1e-9);
            dW[i][j] = 0;
        }
        Vb[i] = alpha1 * Vb[i] + (1 - alpha1) * dB[i];
        Sb[i] = alpha2 * Sb[i] + (1 - alpha2) * dB[i] * dB[i];
        double v = Vb[i] / (1 - alpha1_t);
        double s = Sb[i] / (1 - alpha2_t);
        B[i] -= learningRate * v / (sqrt(s) + 1e-9);
        dB[i] = 0;
    }
    return;
}

void RL::SoftmaxLayer::feedForward(const RL::Vec &x)
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

void RL::SoftmaxLayer::gradient(const RL::Vec &x, const Vec &y)
{
    for (std::size_t i = 0; i < dW.size(); i++) {
        double dy = O[i] - y[i];
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            dW[i][j] += dy * x[j];
        }
        dB[i] += dy;
    }
    return;
}

RL::BPNN &RL::BPNN::operator =(const RL::BPNN &r)
{
    if (this == &r) {
        return *this;
    }
    evalTotalError = r.evalTotalError;
    layers = r.layers;
    return *this;
}

void RL::BPNN::copyTo(BPNN& dstNet)
{
    if (layers.size() != dstNet.layers.size()) {
        return;
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->W.size(); j++) {
            for (std::size_t k = 0; k < layers[i]->W[j].size(); k++) {
                dstNet.layers[i]->W[j][k] = layers[i]->W[j][k];
            }
            dstNet.layers[i]->B[j] = layers[i]->B[j];
        }
    }
    return;
}

void RL::BPNN::softUpdateTo(BPNN &dstNet, double alpha)
{
    if (layers.size() != dstNet.layers.size()) {
        return;
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->W.size(); j++) {
            for (std::size_t k = 0; k < layers[i]->W[j].size(); k++) {
                dstNet.layers[i]->W[j][k] = (1 - alpha) * dstNet.layers[i]->W[j][k] + alpha * layers[i]->W[j][k];
            }
            dstNet.layers[i]->B[j] = (1 - alpha) * dstNet.layers[i]->B[j] + alpha * layers[i]->B[j];
        }
    }
    return;
}

RL::BPNN &RL::BPNN::feedForward(const Vec& x)
{
    layers[0]->feedForward(x);
    for (std::size_t i = 1; i < layers.size(); i++) {
        layers[i]->feedForward(layers[i - 1]->O);
    }
    return *this;
}

RL::Vec& RL::BPNN::output()
{
    return layers.back()->O;
}

double RL::BPNN::gradient(const RL::Vec &x, const RL::Vec &y, RL::BPNN::LossFunc loss)
{
    feedForward(x);
    std::size_t outputIndex = layers.size() - 1;
    loss(layers[outputIndex]->E, layers[outputIndex]->O, y);
    double total = 0;
    if (evalTotalError == true) {
        for (std::size_t i = 0; i < y.size(); i++) {
            total += (layers[outputIndex]->O[i] - y[i])*(layers[outputIndex]->O[i] - y[i]);
        }
        total /= y.size();
    }
    /* error Backpropagate */
    for (int i = outputIndex - 1; i >= 0; i--) {
        layers[i]->backpropagate(layers[i + 1]->E, layers[i + 1]->W);
    }
    /* gradient */
    layers[0]->gradient(x, y);
    for (std::size_t j = 1; j < layers.size(); j++) {
        layers[j]->gradient(layers[j - 1]->O, y);
    }
    return total;
}

void RL::BPNN::SGD(double learningRate)
{
    /* gradient descent */
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->SGD(learningRate);
    }
    return;
}

void RL::BPNN::RMSProp(double rho, double learningRate)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->RMSProp(rho, learningRate);
    }
    return;
}

void RL::BPNN::Adam(double alpha1, double alpha2, double learningRate)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->Adam(alpha1, alpha2, learningRate);
    }
    return;
}

void RL::BPNN::optimize(OptType optType, double learningRate)
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

void RL::BPNN::train(Mat& x,
        Mat& y,
        OptType optType,
        std::size_t batchSize,
        double learningRate,
        std::size_t iterateNum)
{
    if (x.empty() || y.empty()) {
        std::cout<<"x or y is empty"<<std::endl;
        return;
    }
    if (x.size() != y.size()) {
        std::cout<<"x != y"<<std::endl;
        return;
    }
    if (x[0].size() != layers[0]->W[0].size()) {
        std::cout<<"x != w"<<std::endl;
        return;
    }
    Vec &v = layers.back()->O;
    if (y[0].size() != v.size()) {
        std::cout<<"y != output"<<std::endl;
        return;
    }
    int len = x.size();
    for (std::size_t i = 0; i < iterateNum; i++) {
        for (std::size_t j = 0; j < batchSize; j++) {
            int k = rand() % len;
            gradient(x[k], y[k], Loss::CROSS_EMTROPY);
        }
        optimize(optType, learningRate);
    }
    return;
}

int RL::BPNN::argmax()
{
    int index = 0;
    Vec &v = layers.back()->O;
    double maxValue = v[0];
    for (std::size_t i = 0; i < v.size(); i++) {
        if (maxValue < v[i]) {
            maxValue = v[i];
            index = i;
        }
    }
    return index;
}

int RL::BPNN::argmin()
{
    int index = 0;
    Vec &v = layers.back()->O;
    double minValue = v[0];
    for (std::size_t i = 0; i < v.size(); i++) {
        if (minValue > v[i]) {
            minValue = v[i];
            index = i;
        }
    }
    return index;
}

void RL::BPNN::show()
{
    Vec &v = layers.back()->O;
    for (std::size_t i = 0; i < v.size(); i++) {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
    return;
}

void RL::BPNN::load(const std::string& fileName)
{
    std::ifstream file;
    file.open(fileName);
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->W.size(); j++) {
            for (std::size_t k = 0; k < layers[i]->W[j].size(); k++) {
                file >> layers[i]->W[j][k];
            }
            file >> layers[i]->B[j];
        }
    }
    return;
}

void RL::BPNN::save(const std::string& fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->W.size(); j++) {
            for (std::size_t k = 0; k < layers[i]->W[j].size(); k++) {
                file << layers[i]->W[j][k];
            }
            file << layers[i]->B[j];
            file << std::endl;
        }
    }
    return;
}
