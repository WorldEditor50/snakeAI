#ifndef DQNN_H
#define DQNN_H
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
    struct Transition {
        std::vector<float> state;
        float action;
        std::vector<float> nextState;
        float reward;
        bool done;
        Transition(){}
        Transition(std::vector<float>& s, float a,
                   std::vector<float>& s_, float r, bool d)
        {
            state = s;
            action = a;
            nextState = s_;
            reward = r;
            done = d;
        }
    };
    class DQN {
        public:
            DQN(){}
            ~DQN(){}
            void CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           int maxMemorySize = 4096,
                           int replaceTargetIter = 256,
                           int batchSize = 64);
            void Perceive(std::vector<float>& state,
                          float action,
                          std::vector<float>& nextState,
                          float reward,
                          bool done);
            int GreedyAction(std::vector<float>& state);
            int RandomAction();
            int Action(std::vector<float>& state);
            int MaxQ(std::vector<float>& q_value);
            void ExperienceReplay(Transition& x);
            void Learn(int optType = OPT_RMSPROP, float learningRate = 0.001);
            void Save(const std::string& fileName);
            void Load(const std::string& fileName);
            int stateDim;
            int actionDim;
            float gamma;
            float exploringRate;
            int maxMemorySize;
            int batchSize;
            int learningSteps;
            int replaceTargetIter;
            BPNet QMainNet;
            BPNet QTargetNet;
            std::deque<Transition> memories;
    };
}
#endif // DQNN_H
