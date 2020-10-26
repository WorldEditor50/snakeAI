#include "mlp.h"

double ML::Layer::dotProduct(std::vector<double>& x1, std::vector<double>& x2)
{
    double p = 0;
    for (int i = 0; i < x1.size(); i++) {
        p += x1[i] * x2[i];
    }
    return p;
}

void ML::Layer::softmax(std::vector<double>& x, std::vector<double>& y)
{
    double s = 0;
    double maxValue = max(x);
    for (int i = 0; i < x.size(); i++) {
        s += exp(x[i] - maxValue);
    }
    for (int i = 0; i < x.size(); i++) {
        y[i] = exp(x[i] - maxValue) / s;
    }
    return;
}

std::vector<double> ML::Layer::softmax(std::vector<double> &x)
{
    double s = 0;
    double maxValue = max(x);
    for (int i = 0; i < x.size(); i++) {
        s += exp(x[i] - maxValue);
    }
    std::vector<double> y(x.size(), 0);
    for (int i = 0; i < x.size(); i++) {
        y[i] = exp(x[i] - maxValue) / s;
    }
    return y;
}

double ML::Layer::max(std::vector<double> &x)
{
    double value = x[0];
    for (int i = 0; i < x.size(); i++) {
        if (value < x[i]) {
            value = x[i];
        }
    }
    return value;
}

int ML::Layer::argmax(std::vector<double> &x)
{
    int index = 0;
    double value = x[0];
    for (int i = 0; i < x.size(); i++) {
        if (value < x[i]) {
            index = i;
            value = x[i];
        }
    }
    return index;
}

double ML::Layer::activate(double x)
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

double ML::Layer::dActivate(double y)
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

ML::Layer::Layer(int inputDim, int layerDim, int activateType, int lossType, int trainFlag)
{
    if (layerDim < 1 || inputDim < 1) {
        return;
    }
    this->inputDim = inputDim;
    this->layerDim = layerDim;
    this->lossType = lossType;
    this->activateType = activateType;
    W = std::vector<std::vector<double> >(layerDim);
    B = std::vector<double>(layerDim);
    O = std::vector<double>(layerDim);
    E = std::vector<double>(layerDim);
    for (int i = 0; i < W.size(); i++) {
        W[i] = std::vector<double>(inputDim, 0);
    }
    /* buffer for optimization */
    if (trainFlag == 1) {
        dW = std::vector<std::vector<double> >(layerDim);
        dB = std::vector<double>(layerDim);
        Sw = std::vector<std::vector<double> >(layerDim);
        Sb = std::vector<double>(layerDim);
        Vw = std::vector<std::vector<double> >(layerDim);
        Vb = std::vector<double>(layerDim);
        this->alpha1_t = 1;
        this->alpha2_t = 1;
        this->delta = pow(10, -8);
        this->decay = 0;
        for (int i = 0; i < W.size(); i++) {
            dW[i] = std::vector<double>(inputDim);
            Sw[i] = std::vector<double>(inputDim, 0);
            Vw[i] = std::vector<double>(inputDim, 0);
        }
        /* init */
        for (int i = 0; i < W.size(); i++) {
            for (int j = 0; j < W[0].size(); j++) {
                W[i][j] = double(rand() % 10000 - rand() % 10000) / 10000;
            }
            B[i] = double(rand() % 10000 - rand() % 10000) / 10000;
        }
    }
    return;
}

void ML::Layer::feedForward(std::vector<double>& x)
{

    if (x.size() != W[0].size()) {
        std::cout<<"x = "<<x.size()<<std::endl;
        std::cout<<"w = "<<W[0].size()<<std::endl;
        return;
    }
    for (int i = 0; i < W.size(); i++) {
        double y = dotProduct(W[i], x) + B[i];
        O[i] = activate(y);
    }
    if (lossType == LOSS_CROSS_ENTROPY) {
        softmax(O, O);
    }
    return;
}

void ML::Layer::error(std::vector<double>& nextE, std::vector<std::vector<double> >& nextW)
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

void ML::Layer::loss(std::vector<double>& yo, std::vector<double>& yt)
{
    for (int i = 0; i < yo.size(); i++) {
        if (lossType == LOSS_CROSS_ENTROPY) {
            E[i] = -yt[i] * log(yo[i] + 1e-9);
        } else if (lossType == LOSS_KL){
            E[i] = yt[i] * log((yt[i] / (yo[i] + 1e-9)));
        } else if (lossType == LOSS_MSE){
            E[i] = yo[i] - yt[i];
        }
    }
    return;
}

void ML::Layer::loss(std::vector<double> &l)
{
    for (int i = 0; i < l.size(); i++) {
        E[i] = l[i];
    }
    return;
}

void ML::Layer::gradient(std::vector<double>& x)
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

void ML::Layer::softmaxGradient(std::vector<double>& x, std::vector<double>& yo, std::vector<double>& yt)
{
    for (int i = 0; i < dW.size(); i++) {
        double dOutput = yo[i] - yt[i];
        for (int j = 0; j < dW[0].size(); j++) {
            dW[i][j] += dOutput * x[j];
        }
        dB[i] += dOutput;
    }
    return;
}

void ML::Layer::SGD(double learningRate)
{
    /*
     * e = (Activate(wx + b) - T)^2/2
     * de/dw = (Activate(wx +b) - T)*DActivate(wx + b) * x
     * de/db = (Activate(wx +b) - T)*DActivate(wx + b)
     * */
    for (int i = 0; i < W.size(); i++) {
        for (int j = 0; j < W[0].size(); j++) {
            W[i][j] -= learningRate * dW[i][j];
            dW[i][j] = 0;
        }
        B[i] -= learningRate * dB[i];
        dB[i] = 0;
    }
    return;
}

void ML::Layer::RMSProp(double rho, double learningRate)
{
    for (int i = 0; i < W.size(); i++) {
        for (int j = 0; j < W[0].size(); j++) {
            Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
            W[i][j] -= learningRate * dW[i][j] / (sqrt(Sw[i][j]) + delta);
            dW[i][j] = 0;
        }
        Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
        B[i] -= learningRate * dB[i] / (sqrt(Sb[i]) + delta);
        dB[i] = 0;
    }
    return;
}

void ML::Layer::Adam(double alpha1, double alpha2, double learningRate)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (int i = 0; i < W.size(); i++) {
        for (int j = 0; j < W[0].size(); j++) {
            /* momentum */
            Vw[i][j] = alpha1 * Vw[i][j] + (1 - alpha1) * dW[i][j];
            /* delcay factor */
            Sw[i][j] = alpha2 * Sw[i][j] + (1 - alpha2) * dW[i][j] * dW[i][j];
            double v = Vw[i][j] / (1 - alpha1_t);
            double s = Sw[i][j] / (1 - alpha2_t);
            W[i][j] -= learningRate * v / (sqrt(s) + delta);
            dW[i][j] = 0;
        }
        Vb[i] = alpha1 * Vb[i] + (1 - alpha1) * dB[i];
        Sb[i] = alpha2 * Sb[i] + (1 - alpha2) * dB[i] * dB[i];
        double v = Vb[i] / (1 - alpha1_t);
        double s = Sb[i] / (1 - alpha2_t);
        B[i] -= learningRate * v / (sqrt(s) + delta);
        dB[i] = 0;
    }
    return;
}

void ML::Layer::RMSPropWithClip(double rho, double learningRate, double threshold)
{
    /* RMSProp */
    for (int i = 0; i < W.size(); i++) {
        for (int j = 0; j < W[0].size(); j++) {
            Sw[i][j] = rho * Sw[i][j] + (1 - rho) * dW[i][j] * dW[i][j];
            dW[i][j] = dW[i][j] / (sqrt(Sw[i][j]) + delta);
        }
        Sb[i] = rho * Sb[i] + (1 - rho) * dB[i] * dB[i];
        dB[i] = dB[i] / (sqrt(Sb[i]) + delta);
    }
    /* l2 norm of gradient */
    std::vector<double> Wl2(layerDim, 0);
    double bl2 = 0;
    for (int i = 0; i < dW.size(); i++) {
        for (int j = 0; j < dW[0].size(); j++) {
            Wl2[i] += dW[i][j] * dW[i][j];
        }
        bl2 += dB[i] * dB[i];
    }

    for (int i = 0; i < layerDim; i++) {
        Wl2[i] = sqrt(Wl2[i] / layerDim);
    }
    bl2 = sqrt(bl2 / layerDim);
    /* clip gradient */
    for (int i = 0; i < dW.size(); i++) {
        for (int j = 0; j < dW[0].size(); j++) {
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

ML::MLP::MLP(int inputDim, int hiddenDim, int hiddenLayerNum, int outputDim,
             int trainFlag, int activateType, int lossType)
{
    this->lossType = lossType;
    layers.push_back(Layer(inputDim, hiddenDim, activateType, LOSS_MSE, trainFlag));
    for (int i = 1; i < hiddenLayerNum; i++) {
        layers.push_back(Layer(hiddenDim, hiddenDim, activateType, LOSS_MSE, trainFlag));
    }
    if (lossType == LOSS_MSE) {
        layers.push_back(Layer(hiddenDim, outputDim, activateType, LOSS_MSE, trainFlag));
    }
    if (lossType == LOSS_CROSS_ENTROPY || lossType == LOSS_KL) {
        layers.push_back(Layer(hiddenDim, outputDim, ACTIVATE_LINEAR, lossType, trainFlag));
    }
    this->outputIndex = layers.size() - 1;
    return;
}

void ML::MLP::copyTo(MLP& dstNet)
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

void ML::MLP::softUpdateTo(MLP &dstNet, double alpha)
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

int ML::MLP::feedForward(std::vector<double>& x)
{
    layers[0].feedForward(x);
    for (int i = 1; i < layers.size(); i++) {
        layers[i].feedForward(layers[i - 1].O);
    }
    return argmax();
}

std::vector<double>& ML::MLP::getOutput()
{
    std::vector<double>& outputs = layers[outputIndex].O;
    return outputs;
}

void ML::MLP::backPropagate(std::vector<double>& yo, std::vector<double>& yt)
{
    /*  loss */
    layers[outputIndex].loss(yo, yt);
    /* error Backpropagate */
    for (int i = outputIndex - 1; i >= 0; i--) {
        layers[i].error(layers[i + 1].E, layers[i + 1].W);
    }
    return;
}

void ML::MLP::grad(std::vector<double> &x, std::vector<double> &y, std::vector<double> &loss)
{
    /*  loss */
    layers[outputIndex].loss(loss);
    /* error Backpropagate */
    for (int i = outputIndex - 1; i >= 0; i--) {
        layers[i].error(layers[i + 1].E, layers[i + 1].W);
    }
    /* gradient */
    layers[0].gradient(x);
    for (int j = 1; j < layers.size(); j++) {
        if (layers[j].lossType == LOSS_CROSS_ENTROPY || lossType == LOSS_KL) {
            layers[j].softmaxGradient(layers[j - 1].O, layers[outputIndex].O, y);
        } else {
            layers[j].gradient(layers[j - 1].O);
        }
    }
    return;
}

void ML::MLP::gradient(std::vector<double> &x, std::vector<double> &y)
{
    feedForward(x);
    backPropagate(layers[outputIndex].O, y);
    /* gradient */
    layers[0].gradient(x);
    for (int j = 1; j < layers.size(); j++) {
        if (layers[j].lossType == LOSS_CROSS_ENTROPY || lossType == LOSS_KL) {
            layers[j].softmaxGradient(layers[j - 1].O, layers[outputIndex].O, y);
        } else {
            layers[j].gradient(layers[j - 1].O);
        }
    }
    return;
}

void ML::MLP::SGD(double learningRate)
{
    /* gradient descent */
    for (int i = 0; i < layers.size(); i++) {
        layers[i].SGD(learningRate);
    }
    return;
}

void ML::MLP::RMSProp(double rho, double learningRate)
{
    for (int i = 0; i < layers.size(); i++) {
        layers[i].RMSProp(rho, learningRate);
    }
    return;
}

void ML::MLP::Adam(double alpha1, double alpha2, double learningRate)
{
    for (int i = 0; i < layers.size(); i++) {
        layers[i].Adam(alpha1, alpha2, learningRate);
    }
    return;
}

void ML::MLP::RMSPropWithClip(double rho, double learningRate, double threshold)
{
    for (int i = 0; i < layers.size(); i++) {
        layers[i].RMSPropWithClip(rho, learningRate, threshold);
    }
    return;
}

void ML::MLP::optimize(int optType, double learningRate)
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

void ML::MLP::train(std::vector<std::vector<double> >& x,
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
            gradient(x[k], y[k]);
        }
        optimize(optType, learningRate);
    }
    return;
}

int ML::MLP::argmax()
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

void ML::MLP::show()
{
    for (int i = 0; i < layers[outputIndex].O.size(); i++) {
        std::cout<<layers[outputIndex].O[i]<<" ";
    }
    std::cout<<std::endl;
    return;
}

void ML::MLP::load(const std::string& fileName)
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

void ML::MLP::save(const std::string& fileName)
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
