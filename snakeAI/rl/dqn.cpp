#include "dqn.h"

RL::DQN::DQN(std::size_t stateDim, std::size_t hiddenDim, std::size_t actionDim)
{
    this->gamma = 0.99;
    this->exploringRate = 1;
    this->stateDim = stateDim;
    this->actionDim = actionDim;
    this->QMainNet = BPNN(BPNN::Layers{
                              Layer<Sigmoid>::_(stateDim, hiddenDim, true),
                              Layer<Sigmoid>::_(hiddenDim, hiddenDim, true),
                              Layer<Sigmoid>::_(hiddenDim, hiddenDim, true),
                              LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                              Layer<Sigmoid>::_(hiddenDim, actionDim, true)
                          });
    this->QTargetNet = BPNN(BPNN::Layers{
                                 Layer<Sigmoid>::_(stateDim, hiddenDim, false),
                                 Layer<Sigmoid>::_(hiddenDim, hiddenDim, false),
                                 Layer<Sigmoid>::_(hiddenDim, hiddenDim, false),
                                 LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                                 Layer<Sigmoid>::_(hiddenDim, actionDim, false)
                             });
    this->QMainNet.copyTo(QTargetNet);

    return;
}

void RL::DQN::perceive(Vec& state,
        Vec& action,
        Vec& nextState,
        double reward,
        bool done)
{
    if (state.size() != stateDim || nextState.size() != stateDim) {
        return;
    }
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Vec& RL::DQN::sample(Vec &state)
{
    std::uniform_real_distribution<double> distributionReal(0, 1);
    double p = distributionReal(Rand::engine);
    Vec &out = QMainNet.output();
    if (p < exploringRate) {
        out.assign(actionDim, 0);
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        out[index] = 1;
    } else {
        QMainNet.feedForward(state);
    }
    return out;
}

RL::Vec &RL::DQN::output()
{
    return QMainNet.output();
}

int RL::DQN::action(const Vec &state)
{
    int a = QMainNet.feedForward(state).argmax();
    QMainNet.show();
    return a;
}

void RL::DQN::experienceReplay(const Transition& x)
{
    Vec qTarget(actionDim);
    /* estimate q-target: Q-Regression */
    /* select Action to estimate q-value */
    int i = RL::argmax(x.action);
    qTarget = QMainNet.feedForward(x.state).output();
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.feedForward(x.nextState).argmax();
        /* select value in the QTargetNet */
        Vec &v = QTargetNet.feedForward(x.nextState).output();
        qTarget[i] = x.reward + gamma * v[k];
    }
    /* train QMainNet */
    QMainNet.gradient(x.state, qTarget, Loss::MSE);
    return;
}

void RL::DQN::learn(OptType optType,
                    std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    double learningRate)
{
    if (memories.size() < batchSize) {
        return;
    }
    if (learningSteps % replaceTargetIter == 0) {
        std::cout<<"update target net"<<std::endl;
        /* update tagetNet */
        QMainNet.softUpdateTo(QTargetNet, 0.01);
        //QMainNet.copyTo(QTargetNet);
        learningSteps = 0;
    }
    /* experience replay */
    std::uniform_int_distribution<int> distribution(0, memories.size() - 1);
    for (std::size_t i = 0; i < batchSize; i++) {
        int k = distribution(Rand::engine);
        experienceReplay(memories[k]);
    }
    QMainNet.optimize(optType, learningRate);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
    /* update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}

void RL::DQN::save(const std::string &fileName)
{
    QMainNet.save(fileName);
    return;
}

void RL::DQN::load(const std::string &fileName)
{
    QMainNet.load(fileName);
    QMainNet.copyTo(QTargetNet);
    return;
}
