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
#include "mlp.h"
namespace ML {
struct Transition
{
    Vec state;
    Vec action;
    Vec nextState;
    double reward;
    bool done;
    Transition(){}
    explicit Transition(Vec& s, Vec& a,
               Vec& s_, double r, bool d)
    {
        state = s;
        action = a;
        nextState = s_;
        reward = r;
        done = d;
    }
};
class DQN
{
public:
    DQN(){}
    explicit DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t hiddenLayerNum, std::size_t actionDim);
    ~DQN(){}
    void perceive(Vec& state,
                  Vec& action,
                  Vec& nextState,
                  double reward,
                  bool done);
    Vec& greedyAction(Vec& state);
    int randomAction();
    int action(Vec& state);
    int maxQ(Vec& q_value);
    void experienceReplay(Transition& x);
    void learn(OptType optType = OPT_RMSPROP,
               std::size_t maxMemorySize = 4096,
               std::size_t replaceTargetIter = 256,
               std::size_t batchSize = 32,
               double learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
    std::size_t stateDim;
    std::size_t actionDim;
    double gamma;
    double exploringRate;
    int learningSteps;
    MLP QMainNet;
    MLP QTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DQNN_H
