#ifndef NET_HPP
#define NET_HPP
#include "layer.h"

namespace RL {

class Net
{
public:
    using FnLoss = std::function<Tensor(const Tensor&, const Tensor&)>;
    using Layers = std::vector<std::shared_ptr<iLayer> >;
protected:
    float alpha1_t;
    float alpha2_t;
    Layers layers;
public:
    Net(){}
    virtual ~Net(){}
    template<typename ...TLayer>
    explicit Net(TLayer&&...layer):layers({layer...}){}
    explicit Net(const Layers &layers_):layers(layers_){}
    Net(const Net &r):layers(r.layers){}
    void copyTo(Net& dstNet)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            dstNet.layers[i]->w = layers[i]->w;
            dstNet.layers[i]->b = layers[i]->b;
        }
        return;
    }
    void softUpdateTo(Net& dstNet, float alpha)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            lerp(dstNet.layers[i]->w, layers[i]->w, alpha);
            lerp(dstNet.layers[i]->b, layers[i]->b, alpha);
        }
        return;
    }
    Tensor &forward(const Tensor &x)
    {
        layers[0]->forward(x);
        for (std::size_t i = 1; i < layers.size(); i++) {
            Tensor &out = layers[i - 1]->o;
            if ((layers[i - 1]->type == iLayer::LAYER_CONV2D ||
                 layers[i - 1]->type == iLayer::LAYER_MAXPOOLING ||
                 layers[i - 1]->type == iLayer::LAYER_AVGPOOLING)&&
                    layers[i]->type == iLayer::LAYER_FC) {
                layers[i]->forward(out.flatten());
            } else {
                layers[i]->forward(out);
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

    void RMSProp(float rho, float learningRate, float decay)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            layers[i]->RMSProp(rho, learningRate, decay, true);
        }
        return;
    }
    void clamp(float c0, float cn)
    {
        for (std::size_t i = 0; i < layers.size(); i++) {
            Optimize::clamp(layers[i]->w, c0, cn);
            Optimize::clamp(layers[i]->b, c0, cn);
        }
        return;
    }

};

}
#endif // NET_HPP
