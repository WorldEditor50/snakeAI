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
        std::vector<float> state;
        std::vector<float> action;
        float reward;
        Step(){}
        Step(std::vector<float>& s, std::vector<float>& a, float r)
            :state(s), action(a), reward(r) {}
    };
    class DPG {
        public:
            DPG(){}
            ~DPG(){}
            void CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           float learningRate = 0.001f);
            int GreedyAction(std::vector<float>& state);
            int RandomAction();
            int Action(std::vector<float>& state);
            int maxAction(std::vector<float>& value);
            void zscore(std::vector<float>& x);
            void Reinforce(std::vector<Step>& steps, int optType, float learingRate);
            void Save(const std::string& fileName);
            void Load(const std::string& fileName);
            int stateDim;
            int actionDim;
            float gamma;
            float baseLine;
            float exploringRate;
            float learningRate;
            BPNet policyNet;
    };
}
#endif // DPG_H
