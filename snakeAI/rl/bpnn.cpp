#include "bpnn.h"

RL::BPNN &RL::BPNN::operator =(const RL::BPNN &r)
{
    if (this == &r) {
        return *this;
    }
    layers = r.layers;
    return *this;
}

void RL::BPNN::copyTo(BPNN& dstNet)
{
    if (layers.size() != dstNet.layers.size()) {
        return;
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        dstNet.layers[i]->w = layers[i]->w;
        dstNet.layers[i]->b = layers[i]->b;
    }
    return;
}

void RL::BPNN::softUpdateTo(BPNN &dstNet, float alpha)
{
    if (layers.size() != dstNet.layers.size()) {
        return;
    }
    for (std::size_t i = 0; i < layers.size(); i++) {
        dstNet.layers[i]->w = dstNet.layers[i]->w*(1 - alpha) + layers[i]->w*alpha;
        dstNet.layers[i]->b = dstNet.layers[i]->b*(1 - alpha) + layers[i]->b*alpha;
    }
    return;
}

RL::Mat &RL::BPNN::forward(const Mat& x)
{
    layers[0]->forward(x);
    for (std::size_t i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 1]->o);
    }
    return layers.back()->o;
}

RL::Mat &RL::BPNN::output()
{
    return layers.back()->o;
}

void RL::BPNN::backward(const RL::Mat &loss, Mat& E)
{
    std::size_t outputIndex = layers.size() - 1;
    layers[outputIndex]->e = loss;
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    layers[0]->backward(E);
    return;
}

void RL::BPNN::gradient(const RL::Mat &x, const RL::Mat &y)
{
    layers[0]->gradient(x, y);
    for (std::size_t j = 1; j < layers.size(); j++) {
        layers[j]->gradient(layers[j - 1]->o, y);
    }
    return;
}

void RL::BPNN::gradient(const RL::Mat &x, const RL::Mat &y, const RL::BPNN::FnLoss &loss)
{
    forward(x);
    std::size_t outputIndex = layers.size() - 1;
    loss(layers[outputIndex]->e, layers[outputIndex]->o, y);
    /* error Backpropagate */
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    /* gradient */
    layers[0]->gradient(x, y);
    for (std::size_t j = 1; j < layers.size(); j++) {
        layers[j]->gradient(layers[j - 1]->o, y);
    }
    return;
}

void RL::BPNN::gradient(const RL::Mat &x, const RL::Mat &y, const RL::Mat &loss)
{
    std::size_t outputIndex = layers.size() - 1;
    layers[outputIndex]->e = loss;
    /* error Backpropagate */
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    /* gradient */
    layers[0]->gradient(x, y);
    for (std::size_t j = 1; j < layers.size(); j++) {
        layers[j]->gradient(layers[j - 1]->o, y);
    }
    return;
}

void RL::BPNN::SGD(float learningRate)
{
    /* gradient descent */
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->SGD(learningRate);
    }
    return;
}

void RL::BPNN::RMSProp(float rho, float learningRate, float decay)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->RMSProp(rho, learningRate, decay);
    }
    return;
}

void RL::BPNN::NormRMSProp(float rho, float learningRate, float decay)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->NormRMSProp(rho, learningRate, decay);
    }
    return;
}

void RL::BPNN::Adam(float alpha1, float alpha2, float learningRate, float decay)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->Adam(alpha1, alpha2, alpha1_t, alpha2_t, learningRate, decay);
    }
    return;
}

void RL::BPNN::NormAdam(float alpha1, float alpha2, float learningRate, float decay)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->Adam(alpha1, alpha2, alpha1_t, alpha2_t, learningRate, decay);
    }
    return;
}

void RL::BPNN::optimize(OptType optType, float learningRate, float decay)
{
    switch (optType) {
        case OPT_SGD:
            SGD(learningRate);
            break;
        case OPT_RMSPROP:
            RMSProp(0.9, learningRate, decay);
            break;
        case OPT_NORMRMSPROP:
            NormRMSProp(0.9, learningRate, decay);
            break;
        case OPT_ADAM:
            Adam(0.9, 0.99, learningRate, decay);
            break;
        case OPT_NORMADAM:
            NormAdam(0.9, 0.99, learningRate, decay);
            break;
        default:
            RMSProp(0.9, learningRate, decay);
            break;
    }
    return;
}

void RL::BPNN::clamp(float c0, float cn)
{
    for (std::size_t i = 0; i < layers.size(); i++) {
        Optimize::clamp(layers[i]->w, c0, cn);
        Optimize::clamp(layers[i]->b, c0, cn);
    }
    return;
}

void RL::BPNN::show()
{
    Mat &v = layers.back()->o;
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
        for (std::size_t j = 0; j < layers[i]->w.rows; j++) {
            for (std::size_t k = 0; k < layers[i]->w.cols; k++) {
                file >> layers[i]->w(j, k);
            }
            file >> layers[i]->b[j];
        }
    }
    return;
}

void RL::BPNN::save(const std::string& fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (std::size_t i = 0; i < layers.size(); i++) {
        for (std::size_t j = 0; j < layers[i]->w.rows; j++) {
            for (std::size_t k = 0; k < layers[i]->w.cols; k++) {
                file << layers[i]->w(j, k);
            }
            file << layers[i]->b[j];
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
    auto zeta = [](float x, float y) -> float {
        return x*x + y*y + x*y;
    };
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Mat> data;
    std::vector<Mat> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Mat p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Mat q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 128; j++) {
            int k = selectIndex(Random::engine);
            net.forward(data[k]);
            net.gradient(data[k], target[k], Loss::MSE);
        }
        net.Adam();
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Mat x(2, 1);
            float x1 = uniform(Random::engine);
            float x2 = uniform(Random::engine);
            float z = zeta(x1, x2);
            x[0] = x1;
            x[1] = x2;
            Mat &s = net.forward(x);
            std::cout<<"x1 = "<<x1<<" x2 = "<<x2<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
