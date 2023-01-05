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

void RL::BPNN::backward(const RL::Vec &loss, Vec& E)
{
    std::size_t outputIndex = layers.size() - 1;
    layers[outputIndex]->E = loss;
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->E);
    }
    layers[0]->backward(E);
    return;
}

void RL::BPNN::gradient(const RL::Vec &x, const RL::Vec &y)
{
    layers[0]->gradient(x, y);
    for (std::size_t j = 1; j < layers.size(); j++) {
        layers[j]->gradient(layers[j - 1]->O, y);
    }
    return;
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
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->E);
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

void RL::BPNN::RMSProp(double rho, double learningRate, double decay)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->RMSProp(rho, learningRate, decay);
    }
    return;
}

void RL::BPNN::Adam(double alpha1, double alpha2, double learningRate, double decay)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->Adam(alpha1, alpha2, alpha1_t, alpha2_t, learningRate, decay);
    }
    return;
}

void RL::BPNN::optimize(OptType optType, double learningRate, double decay)
{
    switch (optType) {
        case OPT_SGD:
            SGD(learningRate);
            break;
        case OPT_RMSPROP:
            RMSProp(0.9, learningRate, decay);
            break;
        case OPT_ADAM:
            Adam(0.9, 0.99, learningRate, decay);
            break;
        default:
            RMSProp(0.9, learningRate, decay);
            break;
    }
    return;
}

void RL::BPNN::clamp(double c0, double cn)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->W.size(); j++) {
            for (std::size_t k = 0; k < layers[i]->W[j].size(); k++) {
                Optimizer::clamp(layers[i]->W, c0, cn);
            }
            Optimizer::clamp(layers[i]->B, c0, cn);
        }
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

void RL::BPNN::test()
{
    BPNN net(BPNN::Layers{
                 Layer<Sigmoid>::_(2, 16, true),
                 Layer<Sigmoid>::_(16, 16, true),
                 Layer<Sigmoid>::_(16, 16, true),
                 Layer<Linear>::_(16, 1, true)
             });
    auto zeta = [](double x, double y) -> double {
        return x*x + y*y + x*y;
    };
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<Vec> data;
    std::vector<Vec> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Vec p(2);
            double x = uniform(Rand::engine);
            double y = uniform(Rand::engine);
            double z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Vec q(1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 128; j++) {
            int k = selectIndex(Rand::engine);
            net.feedForward(data[k]);
            net.gradient(data[k], target[k], Loss::MSE);
        }
        net.Adam();
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Vec x(2);
            double x1 = uniform(Rand::engine);
            double x2 = uniform(Rand::engine);
            double z = zeta(x1, x2);
            x[0] = x1;
            x[1] = x2;
            auto s = net.feedForward(x).output();
            std::cout<<"x1 = "<<x1<<" x2 = "<<x2<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
