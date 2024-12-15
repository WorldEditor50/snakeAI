#ifndef VAE_HPP
#define VAE_HPP
#include <functional>
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "layer.h"
#include "net.hpp"
#include "loss.h"

namespace RL {

class VAE
{
private:
    int inputDim;
    int zDim;
    Tensor z;
    Tensor eps;
    iLayer::sptr meanLayer;
    iLayer::sptr stdLayer;
    Net encodeNet;
    Net decoder;
public:
    VAE(){}
    explicit VAE(int inputDim_, int hiddenDim, int zDim_)
        :inputDim(inputDim_),zDim(zDim_)
    {
        z = Tensor(zDim, 1);
        eps = Tensor(zDim, 1);
        /* encoder */
        encodeNet = Net(Layer<Tanh>::_(inputDim, hiddenDim, true, true),
                        Layer<Sigmoid>::_(hiddenDim, hiddenDim, true, true));
        meanLayer = Layer<Tanh>::_(hiddenDim, zDim, true, true);
        stdLayer = Layer<Sigmoid>::_(hiddenDim, zDim, true, true);
        /* decoder */
        decoder = Net(Layer<Tanh>::_(zDim, hiddenDim, true, true),
                      Layer<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                      Layer<Sigmoid>::_(hiddenDim, inputDim, true, true));
    }

    void encode(const RL::Tensor &x, Tensor &u, Tensor &std)
    {
        Tensor& o = encodeNet.forward(x);
        u = meanLayer->forward(o, true);
        std = stdLayer->forward(o, true);
        return;
    }

    Tensor& decode(const RL::Tensor &zi)
    {
        return decoder.forward(zi);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false)
    {
        /* encode */
        Tensor& feature = encodeNet.forward(x, inference);
        Tensor& u = meanLayer->forward(feature, inference);
        Tensor& std = stdLayer->forward(feature, inference);
        /*
            gaussian resample
            z = u + std*eps
            eps ~ N(0, 1)
        */
        Random::normal(eps, 0, 1);
        for (std::size_t i = 0; i < z.totalSize; i++) {
            z[i] = u[i] + std[i]*eps[i];
        }
        /* decode */
        return decoder.forward(z, inference);
    }

    void backward(const Tensor& x)
    {
        /* decoder */
        {
            Tensor dLoss = Loss::MSE::df(decoder.output(), x);
            decoder.backward(dLoss);
        }
        /* encoder
           z = u + std*eps
           z -> N(0, 1)
           KL(q(z|x)||p(z)) = -0.5*(1 + log(std*std) - std*std - u*u);
           p(z) ~ N(0, 1)
           dz/du = e + dKL/du = e + u
           dz/ds = eps*(e + dKL/ds) = eps*(e + std - 1/std)
        */
        {
            Tensor& u = meanLayer->o;
            Tensor& std = stdLayer->o;
            Tensor e1(zDim, 1);
            decoder.get(0)->backward(e1);
            for (std::size_t i = 0; i < zDim; i++) {
                meanLayer->e[i] = u[i] + e1[i];
                stdLayer->e[i] = (std[i] - 1.0/(std[i] + 1e-8) + e1[i])*eps[i];
            }
            Tensor& e2 = encodeNet.get(1)->e;
            meanLayer->backward(e2);
            stdLayer->backward(e2);
        }
        /* gradient */
        {
            decoder.gradient(z, x);
            Tensor& o = encodeNet.output();
            meanLayer->gradient(o, z);
            stdLayer->gradient(o, z);
            encodeNet.gradient(x, z);
        }
        return;
    }

    void RMSProp(float lr, float rho=0.9, float decay=0)
    {
        meanLayer->RMSProp(lr, rho, decay, true);
        stdLayer->RMSProp(lr, rho, decay, true);
        encodeNet.RMSProp(lr, rho, decay);
        decoder.RMSProp(lr, rho, decay);
        return;
    }

    void Adam(float lr, float alpha=0.99, float beta=0.9, float decay=0)
    {
        encodeNet.Adam(lr, alpha, beta, decay);
        meanLayer->Adam(lr, alpha, beta, encodeNet.alpha_, encodeNet.beta_, decay, true);
        stdLayer->Adam(lr, alpha, beta, encodeNet.alpha_, encodeNet.beta_, decay, true);
        decoder.Adam(lr, alpha, beta, decay);
        return;
    }

};

}
#endif // VAE_HPP
