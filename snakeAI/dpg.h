#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "mlp.h"
namespace ML {
    struct Step {
        Vec state;
        Vec action;
        double reward;
        Step(){}
        Step(Vec& s, Vec& a, double r)
            :state(s), action(a), reward(r) {}
    };
    class DPG {
        public:
            DPG(){}
            explicit DPG(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim);
            ~DPG(){}
            int greedyAction(Vec& state);
            int randomAction();
            int action(Vec& state);
            int maxAction(Vec& value);
            void zscore(Vec& x);
            void reinforce(OptType optType, double learningRate, std::vector<Step>& x);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
            std::size_t stateDim;
            std::size_t actionDim;
            double gamma;
            double exploringRate;
            double learningRate;
            MLP policyNet;
    };
}
#endif // POLICY_GRADIENT_H
