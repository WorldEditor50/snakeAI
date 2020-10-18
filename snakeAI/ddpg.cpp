#include "ddpg.h"
namespace ML {
    void DDPG::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
            return;
        }
        this->gamma = 0.99;
        this->alpha = 0.01;
        this->beta = 1;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        this->sa.resize(stateDim + actionDim);
        /* actor: a = P(s, theta) */
        this->actorMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
        this->actorTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
        this->actorMainNet.softUpdateTo(actorTargetNet, alpha);
        /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
        this->criticMainNet.createNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 1);
        this->criticTargetNet.createNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 0);
        this->criticMainNet.softUpdateTo(criticTargetNet, alpha);
        return;
    }

    void DDPG::perceive(std::vector<double>& state,
            double action,
            std::vector<double>& nextState,
            double reward,
            bool done)
    {
        if (state.size() != stateDim || nextState.size() != stateDim) {
            return;
        }
        Transition transition;
        transition.state = state;
        transition.action = action;
        transition.nextState = nextState;
        transition.reward = reward;
        transition.done = done;
        memories.push_back(transition);
        return;
    }

    void DDPG::setSA(std::vector<double> &state, std::vector<double> &Action)
    {
        for (int i = 0; i < stateDim; i++) {
            sa[i] = state[i];
        }
        for (int i = stateDim; i < stateDim + actionDim; i++) {
            sa[i] = Action[i];
        }
        return;
    }

    int DDPG::noiseAction(std::vector<double> &state)
    {
        int index = 0;
        actorMainNet.feedForward(state);
        std::vector<double>& out = actorMainNet.getOutput();
        double p = double(rand() % 10000) / 10000;
        if (p < exploringRate) {
            for (int i = 0; i < actionDim; i++) {
                out[i] += double(rand() % 100 - rand() % 100) / 1000;
            }
        }
        index = maxQ(out);
        return index;
    }

    int DDPG::randomAction()
    {
        return rand() % actionDim;
    }

    int DDPG::greedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return 0;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = rand() % actionDim;
        } else {
            index = action(state);
        }
        return index;
    }

    int DDPG::action(std::vector<double> &state)
    {
        int index = 0;
        actorMainNet.feedForward(state);
        std::vector<double>& Action = actorMainNet.getOutput();
        index = maxQ(Action);
        return index;
    }

    int DDPG::maxQ(std::vector<double>& q_value)
    {
        int index = 0;
        double maxValue = q_value[0];
        for (int i = 0; i < q_value.size(); i++) {
            if (maxValue < q_value[i]) {
                maxValue = q_value[i];
                index = i;
            }
        }
        return index;
    }

    void DDPG::experienceReplay(Transition& x)
    {
        std::vector<double> cTarget(actionDim);
        std::vector<double>& aMainOutput = actorMainNet.getOutput();
        std::vector<double>& aTargetOutput = actorTargetNet.getOutput();
        std::vector<double>& cTargetOutput = criticTargetNet.getOutput();
        std::vector<double>& cMainOutput = criticMainNet.getOutput();
        /* estimate Action value */
        int i = int(x.action);
        actorMainNet.feedForward(x.state);
        setSA(x.state, aMainOutput);
        criticMainNet.feedForward(sa);
        cTarget = cMainOutput;
        if (x.done == true) {
            cTarget[i] = x.reward;
        } else {
            actorTargetNet.feedForward(x.nextState);
            setSA(x.nextState, aTargetOutput);
            criticTargetNet.feedForward(sa);
            criticMainNet.feedForward(sa);
            int k = maxQ(cMainOutput);
            cTarget[i] = x.reward + gamma * cTargetOutput[k];
        }
        /* update ActorMainNet */
        std::vector<double> delta(actionDim);
        actorMainNet.feedForward(x.state);
        actorTargetNet.feedForward(x.state);
        setSA(x.state, aMainOutput);
        criticMainNet.feedForward(sa);
        actorMainNet.gradient(x.state, cMainOutput);
        /* update CriticMainNet */
        criticMainNet.gradient(sa, cTarget);
        return;
    }

    void DDPG::learn(int optType, double actorLearningRate, double criticLearningRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            actorMainNet.softUpdateTo(actorTargetNet, alpha);
            criticMainNet.softUpdateTo(criticTargetNet, alpha);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            experienceReplay(memories[k]);
        }
        actorMainNet.optimize(optType, actorLearningRate);
        criticMainNet.optimize(optType, criticLearningRate);
        /* reduce memory */
        if (memories.size() > maxMemorySize) {
            int k = memories.size() / 3;
            for (int i = 0; i < k; i++) {
                memories.pop_front();
            }
        }
        /* update step */
        exploringRate = exploringRate * 0.99999;
        exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
        learningSteps++;
        return;
    }

    void DDPG::save(const std::string& actorPara, const std::string& criticPara)
    {
        actorMainNet.save(actorPara);
        criticMainNet.save(criticPara);
        return;
    }

    void DDPG::load(const std::string& actorPara, const std::string& criticPara)
    {
        actorMainNet.load(actorPara);
        actorMainNet.softUpdateTo(actorTargetNet, alpha);
        criticMainNet.load(criticPara);
        criticMainNet.softUpdateTo(criticTargetNet, alpha);
        return;
    }

}
