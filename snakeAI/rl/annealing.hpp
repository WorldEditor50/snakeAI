#ifndef COSINEANNEALING_HPP
#define COSINEANNEALING_HPP
#include "util.hpp"
#include <iostream>
namespace RL {

class CosineAnnealing
{
public:
    float val;
private:
    float minVal;
    float maxVal;
    int epoch;
    int maxEpoch;
public:
    CosineAnnealing(){}
    explicit CosineAnnealing(float minVal_, float maxVal_, int maxEpoch_)
        :minVal(minVal_),maxVal(maxVal_),epoch(0),maxEpoch(maxEpoch_){}
    float step()
    {
        val = minVal + 0.5*(maxVal - minVal)*(1 + std::cos(pi*epoch/float(maxEpoch)));
        epoch = (epoch + 1)%maxEpoch;
        return val;
    }
};

class ExpAnnealing
{
public:
    float val;
private:
    float minVal;
    float maxVal;
    float r;
public:
    ExpAnnealing(){}
    explicit ExpAnnealing(float minVal_, float maxVal_, float decay=1e-5)
        :val(maxVal_),minVal(minVal_),maxVal(maxVal_),r(1 - decay){}
    float step()
    {
        val *= r;
        val = val < minVal ? minVal : val;
        //std::cout<<"annealing:"<<val<<std::endl;
        return val;
    }
};

class Annealing
{
private:
    float minVal;
    float maxVal;
    float T;
    std::size_t t;
    std::size_t maxT;
public:
    Annealing(){}
    explicit Annealing(float minVal_, float maxVal_, float T_, std::size_t maxT_)
        :minVal(minVal_),maxVal(maxVal_),T(T_),t(0),maxT(maxT_){}
    float step()
    {
        float val = minVal + (maxVal - minVal)*std::tanh(std::exp(-t)*std::cos(t*T));
        t = t < maxT ? t + 1: maxT;
        std::cout<<"annealing:"<<val<<std::endl;
        return val;
    }
};

}
#endif // COSINEANNEALING_HPP
