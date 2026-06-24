#ifndef NET_HPP
#define NET_HPP
#include <memory>
#include <fstream>
#include <functional>
#include "tensor.hpp"
#include "ilayer.h"

namespace RL {

class Net
{
public:
    using FnLoss = std::function<Tensor(const Tensor&, const Tensor&)>;
    using Layers = std::vector<iLayer::sptr>;
    float alpha_;
    float beta_;
protected:
    Layers layers;
public:
    Net():alpha_(1),beta_(1){}
    virtual ~Net(){}
    template<typename ...TLayer>
    explicit Net(TLayer&&...layer)
        :alpha_(1),beta_(1),layers({layer...}){}
    explicit Net(const Layers &layers_)
        :alpha_(1),beta_(1),layers(layers_){}
    Net(const Net &r)
        :alpha_(1),beta_(1),layers(r.layers){}

    inline Tensor& output() {return layers.back()->o;}

    inline iLayer* operator[](std::size_t i) {return layers.at(i).get();}

    inline std::size_t size() const {return layers.size();}

    Tensor &forward(const Tensor &x, bool inference=false)
    {
        layers[0]->forward(x, inference);
        for (std::size_t i = 1; i < layers.size(); i++) {
            Tensor &out = layers[i - 1]->o;
            if ((layers[i - 1]->type == iLayer::LAYER_CONV2D ||
                 layers[i - 1]->type == iLayer::LAYER_MAXPOOLING ||
                 layers[i - 1]->type == iLayer::LAYER_AVGPOOLING)&&
                    layers[i]->type == iLayer::LAYER_FC) {
                layers[i]->forward(out.flatten(), inference);
            } else {
                layers[i]->forward(out, inference);
            }
        }
        return layers.back()->o;
    }

    void backward(const Tensor &x, const Tensor &loss)
    {
        std::size_t outputIndex = layers.size() - 1;
        layers[outputIndex]->e = loss;
        for (int i = layers.size() - 1; i > 0; i--) {
            iLayer::sptr layer = layers[i];
            iLayer::sptr preLayer = layers[i - 1];
            if ((preLayer->type == iLayer::LAYER_CONV2D ||
                 preLayer->type == iLayer::LAYER_MAXPOOLING ||
                 preLayer->type == iLayer::LAYER_AVGPOOLING)&&
                    layer->type == iLayer::LAYER_FC) {
                Tensor e(preLayer->e.totalSize, 1);
                layer->backward(preLayer->o.flatten(), e);
                preLayer->e.val = e.val;
            } else if (preLayer->type == iLayer::LAYER_LSTM) {
                /* LSTM uses BPTT via cacheError/cacheX cache */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            } else if (preLayer->type == iLayer::LAYER_SSM) {
                /* SSM uses the same BPTT pattern as LSTM */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            } else if (preLayer->type == iLayer::LAYER_MAMBA) {
                /* MambaLayer uses the same BPTT pattern as SSM/LSTM */
                Tensor e(preLayer->o.totalSize, 1);
                layer->backward(preLayer->o, e);
                preLayer->cacheError(e);
            } else if ((preLayer->type == iLayer::LAYER_ATTENTION ||
                      preLayer->type == iLayer::LAYER_MHA ||
                      preLayer->type == iLayer::LAYER_SCALEDCONCAT ||
                      preLayer->type == iLayer::LAYER_MOE) &&
                      layer->type == iLayer::LAYER_FC) {
                layer->backward(preLayer->o, preLayer->e);
            } else {
                layer->backward(preLayer->o, preLayer->e);
            }
        }
        /* Also backprop through layer[0] so compound layers
           (TransformerBlock, MHA, etc.) execute their internal
           backward logic, compute LN/MHA gradients, and clear caches.
           Input gradient is discarded (not needed for DQN input). */
        Tensor inputGrad(layers[0]->o.totalSize, 1);
        inputGrad.zero();
        layers[0]->backward(x, inputGrad);
        return;
    }

    void RMSProp(float lr, float rho=0.9, float decay=0)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(lr, rho, decay, true);
        }
        return;
    }

    void Adam(float lr, float alpha=0.99, float beta=0.9, float decay=0)
    {
        alpha_ *= alpha;
        beta_ *= beta;
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->Adam(lr, alpha, beta, alpha_, beta_, decay, true);
        }
        return;
    }

    void clamp(float c0, float cn)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->clamp(c0, cn);
        }
        return;
    }

    void copyTo(Net& dstNet)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->copyTo(dstNet.layers[i].get());
        }
        return;
    }
    void softUpdateTo(Net& dstNet, float alpha)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->softUpdateTo(dstNet.layers[i].get(), alpha);
        }
        return;
    }

    int save(const std::string &fileName) const
    {
        std::ofstream file(fileName);
        if (!file.is_open()) {
            return -1;
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->write(file);
        }
        return 0;
    }

    int load(const std::string &fileName)
    {
        std::ifstream file(fileName);
        if (!file.is_open()) {
            return -1;
        }
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->read(file);
        }
        return 0;
    }
};

}
#endif // NET_HPP
