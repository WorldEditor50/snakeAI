#ifndef VAE_HPP
#define VAE_HPP
#include <memory>
#include "util.hpp"
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
    Net encoder;
    Net decoder;
public:
    VAE(){}
    explicit VAE(int inputDim_, int hiddenDim, int zDim_)
        :inputDim(inputDim_),zDim(zDim_)
    {
        z = Tensor(zDim, 1);
        eps = Tensor(zDim, 1);
        /* encoder */
        encoder = Net(Layer<Tanh>::_(inputDim, hiddenDim, true, true),
                      Layer<Sigmoid>::_(hiddenDim, hiddenDim, true, true));
        meanLayer = Layer<Tanh>::_(hiddenDim, zDim, true, true);
        stdLayer = Layer<Sigmoid>::_(hiddenDim, zDim, true, true);
        /* decoder — Tanh output to match [-1,1] input range (state is normalized) */
        decoder = Net(Layer<Tanh>::_(zDim, hiddenDim, true, true),
                      Layer<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                      Layer<Tanh>::_(hiddenDim, inputDim, true, true));
    }

    void encode(const RL::Tensor &x, Tensor &u, Tensor &std)
    {
        Tensor& o = encoder.forward(x);
        u = meanLayer->forward(o, true);
        std = stdLayer->forward(o, true);
        return;
    }

    Tensor& decode(const RL::Tensor &zi)
    {
        return decoder.forward(zi);
    }

    /* Standard BCQ: sample from prior z ~ N(0, I), decode to get candidate [s_hat, a_hat] */
    Tensor& priorDecode()
    {
        Random::normal(z, 0, 1);  // z ~ N(0, I)
        return decoder.forward(z);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false)
    {
        /* encode */
        Tensor& feature = encoder.forward(x, inference);
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
            decoder.backward(z, dLoss);
        }
        /*
           Reparameterization: z = u + std * eps

           KL(q(z|x)||p(z)) = -0.5*(1 + log(σ²) - σ² - μ²)
           p(z) ~ N(0, 1)

           ∂L_rec/∂u = ∂L_rec/∂z · ∂z/∂u = e1 · 1
           ∂KL/∂u = u

           ∂L_rec/∂σ = ∂L_rec/∂z · ∂z/∂σ = e1 · ε
           ∂KL/∂σ = σ - 1/σ

           ∴ total gradients:
           ∂L/∂u_i = e1_i + u_i
           ∂L/∂σ_i = e1_i * ε_i + (σ_i - 1/σ_i)
        */
        {
            Tensor& u = meanLayer->o;
            Tensor& std = stdLayer->o;
            Tensor e1(zDim, 1);
            Tensor& o = encoder.output();
            decoder[0]->backward(z, e1);
            for (std::size_t i = 0; i < zDim; i++) {
                meanLayer->e[i] = u[i] + e1[i];
                stdLayer->e[i] = e1[i] * eps[i] + (std[i] - 1.0/(std[i] + 1e-8));
            }

            Tensor& e2 = encoder[1]->e;
            meanLayer->backward(o, e2);
            stdLayer->backward(o, e2);
            Tensor encoderGrad(o.shape);
            encoder.backward(x, encoderGrad);
        }

        return;
    }

    void RMSProp(float lr, float rho=0.9, float decay=0)
    {
        meanLayer->RMSProp(lr, rho, decay, true);
        stdLayer->RMSProp(lr, rho, decay, true);
        encoder.RMSProp(lr, rho, decay);
        decoder.RMSProp(lr, rho, decay);
        return;
    }

    void Adam(float lr, float alpha=0.99, float beta=0.9, float decay=0)
    {
        encoder.Adam(lr, alpha, beta, decay);
        meanLayer->Adam(lr, alpha, beta, encoder.alpha_, encoder.beta_, decay, true);
        stdLayer->Adam(lr, alpha, beta, encoder.alpha_, encoder.beta_, decay, true);
        decoder.Adam(lr, alpha, beta, decay);
        return;
    }

};

}
#endif // VAE_HPP
