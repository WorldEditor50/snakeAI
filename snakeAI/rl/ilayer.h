#ifndef ILAYER_H
#define ILAYER_H
#include "tensor.hpp"
#include <memory>
#include <functional>

namespace RL {

class iLayer
{
public:
    enum Type {
        LAYER_FC = 0,
        LAYER_LSTM,
        LAYER_CONCAT,
        LAYER_SCALEDCONCAT,
        LAYER_CONV2D,
        LAYER_MAXPOOLING,
        LAYER_AVGPOOLING,
        LAYER_ATTENTION,
        LAYER_SCALEDDOTPRODUCT
    };
    using sptr = std::shared_ptr<iLayer>;
public:
    int type;
    Tensor o;
    Tensor e;
public:
    iLayer(){}
    virtual ~iLayer(){}
    virtual void initParams(){}
    virtual Tensor& forward(const Tensor& x, bool inference=false)
    {
        return o;
    }
    virtual void gradient(const Tensor& x, const Tensor&){}
    virtual void backward(Tensor &ei){}
    virtual void broadcast(){}
    virtual void cacheError(const Tensor &e){}
    virtual void SGD(float lr){}
    virtual void RMSProp(float lr, float rho, float decay, bool clipGrad){}
    virtual void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad){}
    virtual void clamp(float c0, float cn){}
    virtual void copyTo(iLayer* layer){}
    virtual void softUpdateTo(iLayer* layer, float alpha){}
    virtual void write(std::ofstream &file){}
    virtual void read(std::ifstream &file){}
};

}
#endif // ILAYER_H
