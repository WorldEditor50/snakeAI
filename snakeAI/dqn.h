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
    std::vector<double> state;
    std::vector<double> action;
    std::vector<double> nextState;
    double reward;
    bool done;
    Transition(){}
    explicit Transition(std::vector<double>& s, std::vector<double>& a,
               std::vector<double>& s_, double r, bool d)
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
    explicit DQN(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim);
    ~DQN(){}
    void perceive(std::vector<double>& state,
                  std::vector<double>& action,
                  std::vector<double>& nextState,
                  double reward,
                  bool done);
    std::vector<double>& greedyAction(std::vector<double>& state);
    int randomAction();
    int action(std::vector<double>& state);
    int maxQ(std::vector<double>& q_value);
    void experienceReplay(Transition& x);
    void learn(int optType = OPT_RMSPROP,
               int maxMemorySize = 4096,
               int replaceTargetIter = 256,
               int batchSize = 32,
               double learningRate = 0.001);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
    int stateDim;
    int actionDim;
    double gamma;
    double exploringRate;
    int learningSteps;
    MLP QMainNet;
    MLP QTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DQNN_H
