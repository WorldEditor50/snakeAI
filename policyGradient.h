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
#include "bpnn.h"
namespace ML {
    struct Step {
        std::vector<double> state;
        std::vector<double> action;
        double reward;
    };
    class DPGNet {
        public:
            DPGNet(){}
            ~DPGNet(){}
            void createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           double learningRate = 0.001);
            void perceive(std::vector<double>& state,
                    std::vector<double>& action,
                    double reward);
            int eGreedyAction(std::vector<double>& state);
            int action(std::vector<double>& state);
            int maxQ(std::vector<double>& q_value);
            void normalize(std::vector<double>& x);
            void reinforce(std::vector<Step>& steps);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double exploringRate;
            double learningRate;
            BPNet policyNet;
    };
}
#endif // POLICY_GRADIENT_H
