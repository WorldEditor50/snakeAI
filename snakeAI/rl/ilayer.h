#ifndef ILAYER_H
#define ILAYER_H
#include "tensor.hpp"

namespace RL {

class iLayer
{
public:
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING,
        LAYER_SCALEDDOTPRODUCT
    };
public:
    int type;
    Tensor o;
    Tensor e;
public:
    iLayer(){}
    virtual ~iLayer(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        return o;
    }
    virtual void gradient(const Tensor& x, const Tensor&){}
    virtual void backward(Tensor &ei){}
    virtual void cacheError(const Tensor &e){}
    virtual void SGD(float learningRate){}
    virtual void RMSProp(float rho, float lr, float decay, bool clipGrad){}
    virtual void Adam(float alpha, float beta,
                      float alpha_, float beta_,
                      float lr, float decay, bool clipGrad){}
    virtual void clamp(float c0, float cn){}
    virtual void copyTo(iLayer* layer){}
    virtual void softUpdateTo(iLayer* layer, float alpha){}
};

}
#endif // ILAYER_H
