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
        double action; /* double for continuous action */
        std::vector<double> nextState;
        double reward;
        bool done;
    };
    class DQNet {
        public:
            DQNet(){}
            ~DQNet(){}
            void createNet(int stateDim,
                           int hiddenDim,
                           int hiddenLayerNum,
                           int actionDim,
                           int maxMemorySize = 4096,
                           int replaceTargetIter = 256,
                           int batchSize = 32);
            void perceive(std::vector<double>& state,
                          double action,
                          std::vector<double>& nextState,
                          double reward,
                          bool done);
            void forget();
            int eGreedyAction(std::vector<double>& state);
            int randomAction();
            int action(std::vector<double>& state);
            int maxQ(std::vector<double>& q_value);
            void experienceReplay(Transition& x);
            void learn(int optType = OPT_RMSPROP, double learningRate = 0.001);
            void onlineLearning(std::vector<Transition>& x, int optType, double learningRate);
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
