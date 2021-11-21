#include "bpnn.h"

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
        layers[i]->backward(layers[i + 1]->E, layers[i + 1]->W);
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
        Optimizer::SGD(layers[i]->d.W, layers[i]->W, learningRate);
        Optimizer::SGD(layers[i]->d.B, layers[i]->B, learningRate);
        layers[i]->d.zero();
    }
    return;
}

void RL::BPNN::RMSProp(double rho, double learningRate)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        Optimizer::RMSProp(layers[i]->d.W, layers[i]->s.W, layers[i]->W, learningRate, rho);
        Optimizer::RMSProp(layers[i]->d.B, layers[i]->s.B, layers[i]->B, learningRate, rho);
        layers[i]->d.zero();
    }
    return;
}

void RL::BPNN::Adam(double alpha1, double alpha2, double learningRate)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < layers.size(); i++) {
        Optimizer::Adam(layers[i]->d.W, layers[i]->s.W, layers[i]->v.W, layers[i]->W,
                        alpha1_t, alpha2_t, learningRate, alpha1, alpha2);
        Optimizer::Adam(layers[i]->d.B, layers[i]->s.B, layers[i]->v.B, layers[i]->B,
                        alpha1_t, alpha2_t, learningRate, alpha1, alpha2);
        layers[i]->d.zero();
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
    return RL::argmax(layers.back()->O);
}

int RL::BPNN::argmin()
{
    return RL::argmin(layers.back()->O);
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
