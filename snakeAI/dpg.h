#ifndef DPG_H
#define DPG_H
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
        Step(){}
        Step(std::vector<double>& s, std::vector<double>& a, double r)
            :state(s), action(a), reward(r) {}
    };
    class DPG {
        public:
            DPG(){}
            ~DPG(){}
            void CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           double learningRate = 0.001);
            int GreedyAction(std::vector<double>& state);
            int RandomAction();
            int Action(std::vector<double>& state);
            int maxAction(std::vector<double>& value);
            void zscore(std::vector<double>& x);
            void Reinforce(std::vector<Step>& steps, int optType, double learingRate);
            void Save(const std::string& fileName);
            void Load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double baseLine;
            double exploringRate;
            double learningRate;
            BPNet policyNet;
    };
}
#endif // DPG_H
