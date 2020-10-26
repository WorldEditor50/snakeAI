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
        std::vector<double> state;
        std::vector<double> action;
        double reward;
        Step(){}
        Step(std::vector<double>& s, std::vector<double>& a, double r)
            :state(s), action(a), reward(r) {}
    };
    class DPG {
        public:
            DPG(){}
            explicit DPG(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
            ~DPG(){}
            int greedyAction(std::vector<double>& state);
            int randomAction();
            int action(std::vector<double>& state);
            int maxAction(std::vector<double>& value);
            void zscore(std::vector<double>& x);
            void reinforce(int optType, double learningRate, std::vector<Step>& x);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double exploringRate;
            double learningRate;
            MLP policyNet;
    };
}
#endif // POLICY_GRADIENT_H
