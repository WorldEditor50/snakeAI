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
        double Action; /* double for continuous Action */
        std::vector<double> nextState;
        double reward;
        bool done;
    };
    class DQNet {
        public:
            DQNet(){}
            ~DQNet(){}
            void CreateNet(int stateDim,
                           int hiddenDim,
                           int hiddenLayerNum,
                           int actionDim,
                           int maxMemorySize = 4096,
                           int replaceTargetIter = 256,
                           int batchSize = 32);
            void Perceive(std::vector<double>& state,
                          double Action,
                          std::vector<double>& nextState,
                          double reward,
                          bool done);
            void Forget();
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
