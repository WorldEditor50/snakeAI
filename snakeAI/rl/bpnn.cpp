#include "bpnn.h"

double RL::Layer::dotProduct(Vec& x1, Vec& x2)
{
    double p = 0;
    for (std::size_t i = 0; i < x1.size(); i++) {
        p += x1[i] * x2[i];
    }
    return p;
}

void RL::Layer::softmax(Vec& x, Vec& y)
{
    double s = 0;
    double maxValue = max(x);
    for (std::size_t i = 0; i < x.size(); i++) {
        s += exp(x[i] - maxValue);
    }
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = exp(x[i] - maxValue) / s;
    }
    return;
}

RL::Vec RL::Layer::softmax(Vec &x)
{
    double s = 0;
    double maxValue = max(x);
    for (std::size_t i = 0; i < x.size(); i++) {
        s += exp(x[i] - maxValue);
    }
    Vec y(x.size(), 0);
    for (std::size_t i = 0; i < x.size(); i++) {
        y[i] = exp(x[i] - maxValue) / s;
    }
    return y;
}

double RL::Layer::max(Vec &x)
{
    double value = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (value < x[i]) {
            value = x[i];
        }
    }
    return value;
}

int RL::Layer::argmax(Vec &x)
{
    int index = 0;
    double value = x[0];
    for (std::size_t i = 0; i < x.size(); i++) {
        if (value < x[i]) {
            index = i;
            value = x[i];
        }
    }
    return index;
}

double RL::Layer::activate(double x)
{
    double y = 0;
    switch (activeType) {
        case SIGMOID:
            y = exp(x) / (exp(x) + 1);
            break;
        case RELU:
            y = x > 0 ? x : 0;
            break;
        case TANH:
            y = tanh(x);
            break;
        case LINEAR:
            y = x;
            break;
        default:
            y = exp(x) / (exp(x) + 1);
            break;
    }
    return y;
}

double RL::Layer::dActivate(double y)
{
    double dy = 0;
    switch (activeType) {
        case SIGMOID:
            dy = y * (1 - y);
            break;
        case RELU:
            dy = y  > 0 ? 1 : 0;
            break;
        case TANH:
            dy = 1 - y * y;
            break;
        case LINEAR:
            dy = 1;
            break;
        default:
            dy = y * (1 - y);
            break;
    }
    return dy;
}

RL::Layer::Layer(std::size_t inputDim, std::size_t layerDim, LayerType layerType, ActiveType activeType, LossType lossType, bool trainFlag)
{
    this->inputDim = inputDim;
    this->layerDim = layerDim;
    this->lossType = lossType;
    this->activeType = activeType;
    this->layerType = layerType;
    W = Mat(layerDim);
    B = Vec(layerDim);
    O = Vec(layerDim);
    E = Vec(layerDim);
    for (std::size_t i = 0; i < W.size(); i++) {
        W[i] = Vec(inputDim, 0);
    }
    /* buffer for optimization */
    if (trainFlag == true) {
        dW = Mat(layerDim);
        dB = Vec(layerDim, 0);
        Sw = Mat(layerDim);
        Sb = Vec(layerDim, 0);
        Vw = Mat(layerDim);
        Vb = Vec(layerDim, 0);
        this->alpha1_t = 1;
        this->alpha2_t = 1;
        for (std::size_t i = 0; i < W.size(); i++) {
            dW[i] = Vec(inputDim, 0);
            Sw[i] = Vec(inputDim, 0);
            Vw[i] = Vec(inputDim, 0);
        }
        /* init */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                W[i][j] = double(rand() % 10000 - rand() % 10000) / 10000;
            }
            B[i] = double(rand() % 10000 - rand() % 10000) / 10000;
        }
    }
    return;
}

void RL::Layer::feedForward(Vec& x)
{

    if (x.size() != W[0].size()) {
        std::cout<<"x = "<<x.size()<<std::endl;
        std::cout<<"w = "<<W[0].size()<<std::endl;
        return;
    }
    for (std::size_t i = 0; i < W.size(); i++) {
        double y = dotProduct(W[i], x) + B[i];
        O[i] = activate(y);
    }
    if (lossType == CROSS_ENTROPY) {
        softmax(O, O);
    }
    return;
}

void RL::Layer::error(Vec& nextE, Mat& nextW)
{
    if (E.size() != nextW[0].size()) {
        std::cout<<"size is not matching"<<std::endl;;
    }
    for (std::size_t i = 0; i < nextW[0].size(); i++) {
        for (std::size_t j = 0; j < nextW.size(); j++) {
            E[i] += nextE[j] * nextW[j][i];
        }
    }
    return;
}

void RL::Layer::loss(Vec& yo, Vec& yt)
{
    for (std::size_t i = 0; i < yo.size(); i++) {
        if (lossType == CROSS_ENTROPY) {
            E[i] = -yt[i] * log(yo[i] + 1e-9);
        } else if (lossType == MSE){
            E[i] = yo[i] - yt[i];
        }
    }
    return;
}

void RL::Layer::loss(Vec &l)
{
    for (std::size_t i = 0; i < l.size(); i++) {
        E[i] = l[i];
    }
    return;
}

void RL::Layer::gradient(Vec& x)
{
    for (std::size_t i = 0; i < dW.size(); i++) {
        double dy = dActivate(O[i]);
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            dW[i][j] += E[i] * dy * x[j];
        }
        dB[i] += E[i] * dy;
        E[i] = 0;
    }
    return;
}

void RL::Layer::softmaxGradient(Vec& x, Vec& yo, Vec& yt)
{
    for (std::size_t i = 0; i < dW.size(); i++) {
        double dOutput = yo[i] - yt[i];
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            dW[i][j] += dOutput * x[j];
        }
        dB[i] += dOutput;
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

void RL::Layer::RMSPropWithClip(double rho, double learningRate, double threshold)
{
    /* RMSProp */
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
            dW[i][j] = dW[i][j] / (sqrt(Sw[i][j]) + 1e-9);
        }
        Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
        dB[i] = dB[i] / (sqrt(Sb[i]) + 1e-9);
    }
    /* l2 norm of gradient */
    Vec Wl2(layerDim, 0);
    double bl2 = 0;
    for (std::size_t i = 0; i < dW.size(); i++) {
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            Wl2[i] += dW[i][j] * dW[i][j];
        }
        bl2 += dB[i] * dB[i];
    }

    for (std::size_t i = 0; i < layerDim; i++) {
        Wl2[i] = sqrt(Wl2[i] / layerDim);
    }
    bl2 = sqrt(bl2 / layerDim);
    /* clip gradient */
    for (std::size_t i = 0; i < dW.size(); i++) {
        for (std::size_t j = 0; j < dW[0].size(); j++) {
            if (dW[i][j] * dW[i][j] >= threshold * threshold ) {
                dW[i][j] *= threshold / Wl2[i];
            }
        }
        if (dB[i] * dB[i] >= threshold * threshold) {
            dB[i] *= threshold / bl2;
        }
    }
    /* SGD */
    SGD(learningRate);
    return;
}

RL::BPNN::BPNN(std::size_t inputDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t outputDim,
             bool trainFlag, ActiveType activeType, LossType lossType)
{
    layers.push_back(Layer(inputDim, hiddenDim, Layer::INPUT, activeType, MSE, trainFlag));
    for (std::size_t i = 1; i < hiddenLayerNum; i++) {
        layers.push_back(Layer(hiddenDim, hiddenDim, Layer::HIDDEN, activeType, MSE, trainFlag));
    }
    if (lossType == MSE) {
        layers.push_back(Layer(hiddenDim, outputDim, Layer::OUTPUT, activeType, MSE, trainFlag));
    } else if (lossType == CROSS_ENTROPY) {
        layers.push_back(Layer(hiddenDim, outputDim, Layer::OUTPUT, LINEAR, lossType, trainFlag));
    }
    this->outputIndex = layers.size() - 1;
    return;
}

void RL::BPNN::copyTo(BPNN& dstNet)
{
    if (layers.size() != dstNet.layers.size()) {
        return;
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i].W.size(); j++) {
            for (std::size_t k = 0; k < layers[i].W[j].size(); k++) {
                dstNet.layers[i].W[j][k] = layers[i].W[j][k];
            }
            dstNet.layers[i].B[j] = layers[i].B[j];
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
        for (std::size_t j = 0; j < layers[i].W.size(); j++) {
            for (std::size_t k = 0; k < layers[i].W[j].size(); k++) {
                dstNet.layers[i].W[j][k] = (1 - alpha) * dstNet.layers[i].W[j][k] + alpha * layers[i].W[j][k];
            }
            dstNet.layers[i].B[j] = (1 - alpha) * dstNet.layers[i].B[j] + alpha * layers[i].B[j];
        }
    }
    return;
}

RL::BPNN &RL::BPNN::feedForward(Vec& x)
{
    layers[0].feedForward(x);
    for (std::size_t i = 1; i < layers.size(); i++) {
        layers[i].feedForward(layers[i - 1].O);
    }
    return *this;
}

RL::Vec& RL::BPNN::output()
{
    return layers.back().O;
}

void RL::BPNN::backPropagate(Vec& yo, Vec& yt)
{
    /*  loss */
    layers[outputIndex].loss(yo, yt);
    /* error Backpropagate */
    for (int i = outputIndex - 1; i >= 0; i--) {
        layers[i].error(layers[i + 1].E, layers[i + 1].W);
    }
    return;
}

void RL::BPNN::grad(Vec &x, Vec &y, Vec &loss)
{
    /*  loss */
    layers[outputIndex].loss(loss);
    /* error Backpropagate */
    for (int i = outputIndex - 1; i >= 0; i--) {
        layers[i].error(layers[i + 1].E, layers[i + 1].W);
    }
    /* gradient */
    layers[0].gradient(x);
    for (std::size_t j = 1; j < layers.size(); j++) {
        if (layers[j].lossType == CROSS_ENTROPY) {
            layers[j].softmaxGradient(layers[j - 1].O, layers[outputIndex].O, y);
        } else {
            layers[j].gradient(layers[j - 1].O);
        }
    }
    return;
}

void RL::BPNN::gradient(Vec &x, Vec &y)
{
    feedForward(x);
    backPropagate(layers[outputIndex].O, y);
    /* gradient */
    layers[0].gradient(x);
    for (std::size_t j = 1; j < layers.size(); j++) {
        if (layers[j].lossType == CROSS_ENTROPY) {
            layers[j].softmaxGradient(layers[j - 1].O, layers[outputIndex].O, y);
        } else {
            layers[j].gradient(layers[j - 1].O);
        }
    }
    return;
}

void RL::BPNN::SGD(double learningRate)
{
    /* gradient descent */
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i].SGD(learningRate);
    }
    return;
}

void RL::BPNN::RMSProp(double rho, double learningRate)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i].RMSProp(rho, learningRate);
    }
    return;
}

void RL::BPNN::Adam(double alpha1, double alpha2, double learningRate)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i].Adam(alpha1, alpha2, learningRate);
    }
    return;
}

void RL::BPNN::RMSPropWithClip(double rho, double learningRate, double threshold)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i].RMSPropWithClip(rho, learningRate, threshold);
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
    if (x[0].size() != layers[0].W[0].size()) {
        std::cout<<"x != w"<<std::endl;
        return;
    }
    if (y[0].size() != layers[outputIndex].O.size()) {
        std::cout<<"y != output"<<std::endl;
        return;
    }
    int len = x.size();
    for (std::size_t i = 0; i < iterateNum; i++) {
        for (std::size_t j = 0; j < batchSize; j++) {
            int k = rand() % len;
            gradient(x[k], y[k]);
        }
        optimize(optType, learningRate);
    }
    return;
}

int RL::BPNN::argmax()
{
    int index = 0;
    double maxValue = layers[outputIndex].O[0];
    for (std::size_t i = 0; i < layers[outputIndex].O.size(); i++) {
        if (maxValue < layers[outputIndex].O[i]) {
            maxValue = layers[outputIndex].O[i];
            index = i;
        }
    }
    return index;
}

int RL::BPNN::argmin()
{
    int index = 0;
    double minValue = layers[outputIndex].O[0];
    for (std::size_t i = 0; i < layers[outputIndex].O.size(); i++) {
        if (minValue > layers[outputIndex].O[i]) {
            minValue = layers[outputIndex].O[i];
            index = i;
        }
    }
    return index;
}

void RL::BPNN::show()
{
    for (std::size_t i = 0; i < layers[outputIndex].O.size(); i++) {
        std::cout<<layers[outputIndex].O[i]<<" ";
    }
    std::cout<<std::endl;
    return;
}

void RL::BPNN::load(const std::string& fileName)
{
    std::ifstream file;
    file.open(fileName);
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i].W.size(); j++) {
            for (std::size_t k = 0; k < layers[i].W[j].size(); k++) {
                file >> layers[i].W[j][k];
            }
            file >> layers[i].B[j];
        }
    }
    return;
}

void RL::BPNN::save(const std::string& fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i].W.size(); j++) {
            for (std::size_t k = 0; k < layers[i].W[j].size(); k++) {
                file << layers[i].W[j][k];
            }
            file << layers[i].B[j];
            file << std::endl;
        }
    }
    return;
}
