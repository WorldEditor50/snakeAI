#include <iostream>
#include <string>
#include "rl/vae.hpp"
/* 0 */
RL::Tensor x0({25, 1}, {
               0, 0, 0, 1, 0,
               0, 0, 1, 0, 1,
               0, 0, 1, 0, 1,
               0, 0, 1, 0, 1,
               0, 0, 0, 1, 0});
/* 1 */
RL::Tensor x1({25, 1}, {
               0, 0, 1, 0, 0,
               0, 1, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0});
/* 2 */
RL::Tensor x2({25, 1}, {
               1, 1, 1, 1, 0,
               0, 0, 0, 1, 0,
               0, 0, 1, 0, 0,
               0, 1, 0, 0, 0,
               1, 1, 1, 1, 0});
/* 3 */
RL::Tensor x3({25, 1}, {
               0, 0, 1, 1, 0,
               0, 0, 0, 0, 1,
               0, 0, 0, 1, 0,
               0, 0, 0, 0, 1,
               0, 0, 1, 1, 0
          });
/* 4 */
RL::Tensor x4({25, 1}, {
               0, 1, 0, 1, 0,
               0, 1, 0, 1, 0,
               0, 1, 0, 1, 0,
               0, 1, 1, 1, 1,
               0, 0, 0, 1, 0});
/* 5 */
RL::Tensor x5({25, 1}, {
               0, 1, 1, 1, 1,
               0, 1, 0, 0, 0,
               0, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               0, 1, 1, 1, 0});
/* 6 */
RL::Tensor x6({25, 1}, {
               0, 1, 1, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 0, 0,
               1, 0, 0, 1, 0,
               0, 1, 1, 1, 0});
/* 7 */
RL::Tensor x7({25, 1}, {
               0, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               0, 0, 0, 1, 0,
               0, 0, 0, 1, 0,
               0, 0, 0, 1, 0});
/* 8 */
RL::Tensor x8({25, 1}, {
               0, 0, 1, 0, 0,
               0, 1, 0, 1, 0,
               0, 0, 1, 0, 0,
               0, 1, 0, 1, 0,
               0, 0, 1, 0, 0});
/* 9 */
RL::Tensor x9({25, 1}, {
               0, 1, 0, 0, 0,
               1, 0, 1, 0, 0,
               0, 1, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0});
void print(const RL::Tensor &x)
{
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (x(i, j) > 0.5) {
                std::cout<<"o";
            } else {
                std::cout<<" ";
            }
        }
        std::cout<<std::endl;
    }
    return;
}
void test_vae()
{
    RL::VAE vae(25, 64, 10);
    RL::Tensor x[10] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9};
    std::uniform_int_distribution<int> uniform(0, 9);
    for (int i = 0; i < 3000; i++) {
        for (int j = 0; j < 32; j++) {
            int k = uniform(RL::Random::generator);
            vae.forward(x[k]);
            vae.backward(x[k]);
        }
        vae.RMSProp(1e-2);
    }
    for (int i = 0; i < 10; i++) {
        RL::Tensor xh = vae.forward(x[i]);
        xh.reshape(5, 5);
        print(xh);
        std::cout<<"======="<<std::endl;
    }
    return;
}

void test_ae()
{
    /* auto encoder */
    int inputDim = 25;
    int hiddenDim = 64;
    int zDim = 10;
    RL::Net ae(RL::Layer<RL::Sigmoid>::_(inputDim, hiddenDim, true, true),
               RL::Layer<RL::Sigmoid>::_(hiddenDim, hiddenDim, true, true),
               RL::Layer<RL::Tanh>::_(hiddenDim, zDim, true, true),
               RL::Layer<RL::Sigmoid>::_(zDim, hiddenDim, true, true),
               RL::Layer<RL::Sigmoid>::_(hiddenDim, inputDim, true, true));
    RL::Tensor x[10] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9};
    std::uniform_int_distribution<int> uniform(0, 9);
    for (int i = 0; i < 3000; i++) {
        for (int j = 0; j < 32; j++) {
            int k = uniform(RL::Random::generator);
            RL::Tensor& out = ae.forward(x[k]);
            RL::Tensor loss = RL::Loss::MSE::df(out, x[k]);
            ae.backward(loss);
            ae.gradient(x[k], x[k]);
        }
        ae.RMSProp(1e-2);
    }
    for (int i = 0; i < 10; i++) {
        RL::Tensor xh = ae.forward(x[i]);
        xh.reshape(5, 5);
        print(xh);
        std::cout<<"======="<<std::endl;
    }
    return;
}

int main()
{
    //test_ae();
    test_vae();
    return 0;
}
