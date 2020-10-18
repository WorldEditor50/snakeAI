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
            void createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
                           int maxMemorySize = 4096,
                           int replaceTargetIter = 256,
                           int batchSize = 32);
            void perceive(std::vector<double>& state,
                          double action,
                          std::vector<double>& nextState,
                          double reward,
                          bool done);
            int greedyAction(std::vector<double>& state);
            int randomAction();
            int action(std::vector<double>& state);
            int maxQ(std::vector<double>& q_value);
            void experienceReplay(Transition& x);
            void learn(int optType = OPT_RMSPROP, double learningRate = 0.001);
            void save(const std::string& fileName);
            void load(const std::string& fileName);
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
