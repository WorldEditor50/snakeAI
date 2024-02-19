#include "dqn.h"

RL::DQN::DQN(std::size_t stateDim_, std::size_t hiddenDim, std::size_t actionDim_)
{
    gamma = 0.99;
    exploringRate = 1;
    totalReward = 0;
    stateDim = stateDim_;
    actionDim = actionDim_;
    QMainNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, true),
                    LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                    Layer<Tanh>::_(hiddenDim, hiddenDim, true),
                    LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, true),
                    Layer<Sigmoid>::_(hiddenDim, actionDim, true));

    QTargetNet = BPNN(Layer<Tanh>::_(stateDim, hiddenDim, false),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                      Layer<Tanh>::_(hiddenDim, hiddenDim, false),
                      LayerNorm<Sigmoid>::_(hiddenDim, hiddenDim, false),
                      Layer<Sigmoid>::_(hiddenDim, actionDim, false));

    QMainNet.copyTo(QTargetNet);
}

void RL::DQN::perceive(const Mat& state,
                       const Mat& action,
                       const Mat& nextState,
                       float reward,
                       bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    return;
}

RL::Mat& RL::DQN::eGreedyAction(const Mat &state)
{
    std::uniform_real_distribution<float> distributionReal(0, 1);
    float p = distributionReal(Rand::engine);
    Mat &out = QMainNet.output();
    if (p < exploringRate) {
        out.zero();
        std::uniform_int_distribution<int> distribution(0, actionDim - 1);
        int index = distribution(Rand::engine);
        out[index] = 1;
    } else {
        QMainNet.forward(state);
    }
    return out;
}

RL::Mat &RL::DQN::output()
{
    return QMainNet.output();
}

int RL::DQN::action(const Mat &state)
{
    int a = QMainNet.forward(state).argmax();
    QMainNet.show();
    return a;
}

void RL::DQN::experienceReplay(const Transition& x)
{
    /* estimate q-target: Q-Regression */
    /* select Action to estimate q-value */
    int i = x.action.argmax();
    Mat qTarget = QMainNet.forward(x.state);
    if (x.done == true) {
        qTarget[i] = x.reward;
    } else {
        /* select optimal Action in the QMainNet */
        int k = QMainNet.forward(x.nextState).argmax();
        /* select value in the QTargetNet */
        Mat &v = QTargetNet.forward(x.nextState);
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
                    float learningRate)
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
    QMainNet.optimize(optType, learningRate, 0);
    /* reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 3;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
        }
    }
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
