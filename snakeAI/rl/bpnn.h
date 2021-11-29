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
    using LossFunc = std::function<void(Vec&, const Vec&, const Vec&)>;
    using Layers = std::vector<std::shared_ptr<LayerObject> >;
public:
    BPNN(){}
    virtual ~BPNN(){}
    explicit BPNN(const Layers &layers_):evalTotalError(false),layers(layers_){}
    BPNN(const BPNN &r):evalTotalError(r.evalTotalError),layers(r.layers){}
    BPNN &operator = (const BPNN &r);
    void copyTo(BPNN& dstNet);
    void softUpdateTo(BPNN& dstNet, double alpha);
    Vec& output();
    BPNN &feedForward(const Vec &x);
    double gradient(const Vec &x, const Vec &y, LossFunc loss);
    void SGD(double learningRate = 0.001);
    void RMSProp(double rho = 0.9, double learningRate = 0.001);
    void Adam(double alpha1 = 0.9, double alpha2 = 0.99, double learningRate = 0.001);
    void optimize(OptType optType = OPT_RMSPROP, double learningRate = 0.001);
    int argmax();
    int argmin();
    void show();
    void load(const std::string& fileName);
    void save(const std::string& fileName);
    static void test();
protected:
    double alpha1_t;
    double alpha2_t;
    bool evalTotalError;
    Layers layers;
};

}
#endif // BPNN_H
