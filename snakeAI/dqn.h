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
        std::vector<double> state;
        double action;
        std::vector<double> nextState;
        double reward;
        bool done;
        Transition(){}
        Transition(std::vector<double>& s, double a,
                   std::vector<double>& s_, double r, bool d)
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
            void Perceive(std::vector<double>& state,
                          double action,
                          std::vector<double>& nextState,
                          double reward,
                          bool done);
            int GreedyAction(std::vector<double>& state);
            int RandomAction();
            int Action(std::vector<double>& state);
            int MaxQ(std::vector<double>& q_value);
            void ExperienceReplay(Transition& x);
            void Learn(int optType = OPT_RMSPROP, double learningRate = 0.001);
            void Save(const std::string& fileName);
            void Load(const std::string& fileName);
            int stateDim;
            int actionDim;
            double gamma;
            double exploringRate;
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
