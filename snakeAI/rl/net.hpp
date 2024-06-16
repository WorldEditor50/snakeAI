#ifndef NET_HPP
#define NET_HPP
#include <memory>
#include "tensor.hpp"
#include "ilayer.h"

namespace RL {

class Net
{
public:
    struct Connection {
        int from;
        int to;
    };
    using FnLoss = std::function<Tensor(const Tensor&, const Tensor&)>;
    using Layers = std::vector<std::shared_ptr<iLayer> >;
protected:
    float alpha_;
    float beta_;
    std::vector<Connection> connections;
    Layers layers;
public:
    Net():alpha_(1),beta_(1){}
    virtual ~Net(){}
    template<typename ...TLayer>
    explicit Net(TLayer&&...layer)
        :layers({layer...}),alpha_(1),beta_(1){}
    explicit Net(const Layers &layers_)
        :layers(layers_),alpha_(1),beta_(1){}
    Net(const Net &r)
        :layers(r.layers),connections(r.connections),alpha_(1),beta_(1){}

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

    void backward(const Tensor &loss)
    {
        std::size_t outputIndex = layers.size() - 1;
        layers[outputIndex]->e = loss;
        for (int i = layers.size() - 1; i > 0; i--) {
            if ((layers[i - 1]->type == iLayer::LAYER_CONV2D ||
                 layers[i - 1]->type == iLayer::LAYER_MAXPOOLING ||
                 layers[i - 1]->type == iLayer::LAYER_AVGPOOLING)&&
                    layers[i]->type == iLayer::LAYER_FC) {
                Tensor e(layers[i - 1]->e.totalSize, 1);
                layers[i]->backward(e);
                layers[i - 1]->e.val = e.val;
            } else if(layers[i - 1]->type == iLayer::LAYER_LSTM &&
                      layers[i]->type == iLayer::LAYER_FC) {
                Tensor e(layers[i - 1]->o.totalSize, 1);
                layers[i]->backward(e);
                layers[i - 1]->cacheError(e);
            } else {
                layers[i]->backward(layers[i - 1]->e);
            }
        }
        return;
    }
    void gradient(const Tensor &x, const Tensor &y)
    {
        layers[0]->gradient(x, y);
        for (std::size_t i = 1; i < layers.size(); i++) {
            Tensor &out = layers[i - 1]->o;
            if ((layers[i - 1]->type == iLayer::LAYER_CONV2D ||
                 layers[i - 1]->type == iLayer::LAYER_MAXPOOLING ||
                 layers[i - 1]->type == iLayer::LAYER_AVGPOOLING)&&
                    layers[i]->type == iLayer::LAYER_FC) {
                layers[i]->gradient(out.flatten(), y);
            } else {
                layers[i]->gradient(out, y);
            }

        }
        return;
    }

    void RMSProp(float lr, float rho, float decay)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(lr, rho, decay, true);
        }
        return;
    }

    void Adam(float lr, float alpha=0.99, float beta=0.9, float decay=0)
    {
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
};

}
#endif // NET_HPP
