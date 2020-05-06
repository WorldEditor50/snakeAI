#include "ddpg.h"
namespace ML {
    void DDPGNet::createNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
            return;
        }
        this->gamma = 0.99;
        this->alpha = 0.1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        /* actor: a = P(s, theta) */
        this->ActorMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
        this->ActorTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID, LOSS_CROSS_ENTROPY);
        this->ActorMainNet.softUpdateTo(ActorTargetNet, alpha);
        /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
        this->CriticMainNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID);
        this->CriticTargetNet.createNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, ACTIVATE_SIGMOID);
        this->CriticMainNet.softUpdateTo(CriticTargetNet, alpha);
        return;
    }

    void DDPGNet::perceive(std::vector<double>& state,
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

    void DDPGNet::forget()
    {
        int k = memories.size() / 3;
        for (int i = 0; i < k; i++) {
            memories.pop_front();
        }
        return;
    }

    int DDPGNet::action(std::vector<double> &state)
    {
        int index = 0;
        ActorMainNet.feedForward(state);
        std::vector<double>& action = ActorMainNet.getOutput();
        index = maxQ(action);
        return index;
    }

    int DDPGNet::maxQ(std::vector<double>& q_value)
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

    void DDPGNet::experienceReplay(Transition& x)
    {
        std::vector<double> cTarget(actionDim);
        std::vector<double>& aTargetOutput = ActorTargetNet.getOutput();
        std::vector<double>& cTargetOutput = CriticTargetNet.getOutput();
        std::vector<double>& cMainOutput = CriticMainNet.getOutput();
        /* estimate action value */
        int i = int(x.action);
        CriticMainNet.feedForward(x.state);
        cTarget = cMainOutput;
        if (x.done == true) {
            cTarget[i] = x.reward;
        } else {
            CriticTargetNet.feedForward(x.nextState);
            CriticMainNet.feedForward(x.nextState);
            int k = maxQ(cMainOutput);
            cTarget[i] = x.reward + gamma * cTargetOutput[k];
        }
        /* update ActorMainNet */
        ActorMainNet.calculateGradient(x.state, cMainOutput);
        /* update CriticMainNet */
        CriticMainNet.calculateGradient(x.state, cTarget);
        return;
    }

    void DDPGNet::learn(int optType, double actorLearningRate, double criticLearningRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            ActorMainNet.softUpdateTo(ActorTargetNet, alpha);
            CriticMainNet.softUpdateTo(CriticTargetNet, alpha);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            experienceReplay(memories[k]);
        }
        ActorMainNet.optimize(optType, actorLearningRate);
        CriticMainNet.optimize(optType, criticLearningRate);
        /* reduce memory */
        if (memories.size() > maxMemorySize) {
            forget();
        }
        /* update step */
        learningSteps++;
        return;
    }

    void DDPGNet::save(const std::string& actorPara, const std::string& criticPara)
    {
        ActorMainNet.save(actorPara);
        CriticMainNet.save(criticPara);
        return;
    }

    void DDPGNet::load(const std::string& actorPara, const std::string& criticPara)
    {
        ActorMainNet.load(actorPara);
        ActorMainNet.softUpdateTo(ActorTargetNet, alpha);
        CriticMainNet.load(criticPara);
        CriticMainNet.softUpdateTo(CriticTargetNet, alpha);
        return;
    }
}
