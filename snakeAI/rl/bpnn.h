#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <functional>
#include <memory>
#include <tuple>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "layer.h"

namespace RL {

/* Optimize method */
enum OptType {
    NONE = 0,
    OPT_SGD,
    OPT_RMSPROP,
    OPT_ADAM
};

class BPNN
{
public:
    using FnLoss = std::function<void(Mat&, const Mat&, const Mat&)>;
    using Layers = std::vector<std::shared_ptr<LayerObject> >;
protected:
    float alpha1_t;
    float alpha2_t;
public:
    Layers layers;
public:
    BPNN(){}
    virtual ~BPNN(){}
    template<typename ...TLayer>
    explicit BPNN(TLayer&&...layer):layers({layer...}){}
    explicit BPNN(const Layers &layers_):layers(layers_){}
    BPNN(const BPNN &r):layers(r.layers){}
    BPNN &operator = (const BPNN &r);
    void copyTo(BPNN& dstNet);
    void softUpdateTo(BPNN& dstNet, float alpha);
    Mat &forward(const Mat &x);
    inline Mat& operator()(const Mat &x) {return forward(x);}
    Mat &output();
    void backward(const Mat &loss, Mat& E);
    void gradient(const Mat &x, const Mat &y);
    void gradient(const Mat &x, const Mat &y, const FnLoss &loss);
    void gradient(const Mat &x, const RL::Mat &y, const Mat &loss);
    void SGD(float learningRate = 0.001);
    void RMSProp(float rho = 0.9, float learningRate = 0.001, float decay = 0);
    void Adam(float alpha1 = 0.9, float alpha2 = 0.99, float learningRate = 0.001, float decay = 0);
    void optimize(OptType optType = OPT_RMSPROP, float learningRate = 0.001, float decay = 0);
    void clamp(float c0, float cn);
    void show();
    void load(const std::string& fileName);
    void save(const std::string& fileName);
    static void test();

};

}
#endif // BPNN_H
