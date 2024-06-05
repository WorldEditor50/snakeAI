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
        lerp(dstNet.layers[i]->w, layers[i]->w, alpha);
        lerp(dstNet.layers[i]->b, layers[i]->b, alpha);
    }
    return;
}

RL::Tensor &RL::BPNN::forward(const Tensor& x)
{
    layers[0]->forward(x);
    for (std::size_t i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 1]->o);
    }
    return layers.back()->o;
}

RL::Tensor &RL::BPNN::output()
{
    return layers.back()->o;
}

void RL::BPNN::backward(const RL::Tensor &loss, Tensor& e)
{
    std::size_t outputIndex = layers.size() - 1;
    layers[outputIndex]->e = loss;
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    layers[0]->backward(e);
    return;
}

void RL::BPNN::backward(const RL::Tensor &loss)
{
    std::size_t outputIndex = layers.size() - 1;
    layers[outputIndex]->e = loss;
    /* error Backpropagate */
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    return;
}

void RL::BPNN::gradient(const RL::Tensor &x, const RL::Tensor &y)
{
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
        layers[i]->RMSProp(rho, learningRate, decay, true);
    }
    return;
}


void RL::BPNN::Adam(float alpha1, float alpha2, float learningRate, float decay)
{
    alpha1_t *= alpha1;
    alpha2_t *= alpha2;
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->Adam(alpha1, alpha2, alpha1_t, alpha2_t, learningRate, decay, true);
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
        case OPT_ADAM:
            Adam(0.9, 0.99, learningRate, decay);
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
    Tensor &v = layers.back()->o;
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
        for (int j = 0; j < layers[i]->w.shape[0]; j++) {
            for (int k = 0; k < layers[i]->w.shape[1]; k++) {
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
        for (int j = 0; j < layers[i]->w.shape[0]; j++) {
            for (int k = 0; k < layers[i]->w.shape[1]; k++) {
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
    std::vector<Tensor> data;
    std::vector<Tensor> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Tensor q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 128; j++) {
            int k = selectIndex(Random::engine);
            Tensor& out = net.forward(data[k]);
            net.backward(Loss::MSE(out, target[k]));
            net.gradient(data[k], target[k]);
        }
        net.Adam();
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Tensor x(2, 1);
            float x1 = uniform(Random::engine);
            float x2 = uniform(Random::engine);
            float z = zeta(x1, x2);
            x[0] = x1;
            x[1] = x2;
            Tensor &s = net.forward(x);
            std::cout<<"x1 = "<<x1<<" x2 = "<<x2<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
