#ifndef DDPG_H
#define DDPG_H
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

/* TODO:
 * 1. building critic network: Q(S, A, α, β) = V(S, α) + A(S, A, β)
 * 2. apply DDPG to discrete action and figure out a loss function
 * 3. complete experience-replay
 */
class DDPGNet
{
public:
    DDPGNet(){}
    ~DDPGNet(){}
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
    int action(std::vector<double>& state);
    int maxQ(std::vector<double>& q_value);
    void experienceReplay(Transition& x);
    void learn(int optType = OPT_RMSPROP,
               double actorLearningRate = 0.0001,
               double criticLearningRate = 0.001);
    void save(const std::string& actorPara, const std::string& criticPara);
    void load(const std::string& actorPara, const std::string& criticPara);
    int stateDim;
    int actionDim;
    double gamma;
    double alpha;
    int maxMemorySize;
    int batchSize;
    int learningSteps;
    int replaceTargetIter;
    BPNet ActorMainNet;
    BPNet ActorTargetNet;
    BPNet CriticMainNet;
    BPNet CriticTargetNet;
    std::deque<Transition> memories;
};
}
#endif // DDPG_H
