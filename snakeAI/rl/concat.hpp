#ifndef CONCAT_HPP
#define CONCAT_HPP
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"

namespace RL {

template<typename TLayer, typename Fn, int N>
class Concat : public iLayer
{
public:
    Tensor w1;
    Tensor w2;
    Tensor b;
    TLayer layers[N];
public:
    Concat(){}
    explicit Concat(std::size_t inputDim, std::size_t outputDim, bool bias_, bool withGrad_)
    {

    }

    static std::shared_ptr<Concat> _(std::size_t inputDim,
                                     std::size_t outputDim,
                                     bool bias,
                                     bool withGrad)
    {
        return std::make_shared<Concat>(inputDim, outputDim, bias, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        return o;
    }

    void backward(Tensor &ei) override
    {

        return;
    }

    void gradient(const Tensor& x, const Tensor&) override
    {

        return;
    }
};
}
#endif // CONCAT_HPP
