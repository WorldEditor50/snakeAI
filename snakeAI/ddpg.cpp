#include "ddpg.h"
namespace ML {
    void DDPG::CreateNet(int stateDim, int hiddenDim, int hiddenLayerNum, int actionDim,
            int maxMemorySize, int replaceTargetIter, int batchSize)
    {
        if (stateDim < 1 || hiddenDim < 1 || hiddenLayerNum < 1 || actionDim < 1 ||
                maxMemorySize < 1 || replaceTargetIter < 1 || batchSize < 1) {
            return;
        }
        this->gamma = 0.99;
        this->alpha = 0.01f;
        this->beta = 1;
        this->exploringRate = 1;
        this->stateDim = stateDim;
        this->actionDim = actionDim;
        this->maxMemorySize = maxMemorySize;
        this->replaceTargetIter = replaceTargetIter;
        this->batchSize = batchSize;
        this->sa.resize(stateDim + actionDim);
        /* actor: a = P(s, theta) */
        this->actorMainNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_KL);
        this->actorTargetNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_KL);
        this->actorMainNet.SoftUpdateTo(actorTargetNet, alpha);
        /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
        this->criticMainNet.CreateNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 1);
        this->criticTargetNet.CreateNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 0);
        this->criticMainNet.SoftUpdateTo(criticTargetNet, alpha);
        return;
    }

    void DDPG::Perceive(std::vector<float>& state,
            float action,
            std::vector<float>& nextState,
            float reward,
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

    void DDPG::SetSA(std::vector<float> &state, std::vector<float> &action)
    {
        for (int i = 0; i < stateDim; i++) {
            sa[i] = state[i];
        }
        for (int i = stateDim; i < stateDim + actionDim; i++) {
            sa[i] = action[i];
        }
        return;
    }

    int DDPG::NoiseAction(std::vector<float> &state)
    {
        int index = 0;
        actorMainNet.FeedForward(state);
        std::vector<float>& Action = actorMainNet.GetOutput();
        for (int i = 0; i < actionDim; i++) {
            Action[i] += float(rand() % 100 - rand() % 100) / 1000;
        }
        index = MaxQ(Action);
        return index;
    }

    int DDPG::RandomAction()
    {
        return rand() % actionDim;
    }

    int DDPG::GreedyAction(std::vector<float> &state)
    {
        if (state.size() != stateDim) {
            return 0;
        }
        float p = float(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = rand() % actionDim;
        } else {
            index = actorMainNet.FeedForward(state);
        }
        return index;
    }

    int DDPG::Action(std::vector<float> &state)
    {
        return actorMainNet.FeedForward(state);
    }

    int DDPG::MaxQ(std::vector<float>& q_value)
    {
        int index = 0;
        float maxValue = q_value[0];
        for (int i = 0; i < q_value.size(); i++) {
            if (maxValue < q_value[i]) {
                maxValue = q_value[i];
                index = i;
            }
        }
        return index;
    }

    void DDPG::ExperienceReplay(Transition& x)
    {
        std::vector<float> cTarget(actionDim);
        std::vector<float>& aMainOutput = actorMainNet.GetOutput();
        std::vector<float>& aTargetOutput = actorTargetNet.GetOutput();
        std::vector<float>& cTargetOutput = criticTargetNet.GetOutput();
        std::vector<float>& cMainOutput = criticMainNet.GetOutput();
        /* estimate Action value */
        int i = int(x.action);
        actorMainNet.FeedForward(x.state);
        SetSA(x.state, aMainOutput);
        criticMainNet.FeedForward(sa);
        cTarget = cMainOutput;
        if (x.done == true) {
            cTarget[i] = x.reward;
        } else {
            actorTargetNet.FeedForward(x.nextState);
            SetSA(x.nextState, aTargetOutput);
            criticTargetNet.FeedForward(sa);
            criticMainNet.FeedForward(sa);
            int k = MaxQ(cMainOutput);
            cTarget[i] = x.reward + gamma * cTargetOutput[k];
        }
        /* update actorMainNet */
        std::vector<float> delta(actionDim);
        actorMainNet.FeedForward(x.state);
        actorTargetNet.FeedForward(x.state);
        SetSA(x.state, aMainOutput);
        criticMainNet.FeedForward(sa);
        actorMainNet.Gradient(x.state, cMainOutput);
        /* update criticMainNet */
        criticMainNet.Gradient(sa, cTarget);
        return;
    }

    void DDPG::Learn(int optType, float actorLearningRate, float criticLearningRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            actorMainNet.SoftUpdateTo(actorTargetNet, alpha);
            criticMainNet.SoftUpdateTo(criticTargetNet, alpha);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            ExperienceReplay(memories[k]);
        }
        actorMainNet.Optimize(optType, actorLearningRate);
        criticMainNet.Optimize(optType, criticLearningRate);
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

    void DDPG::Save(const std::string& actorPara, const std::string& criticPara)
    {
        actorMainNet.Save(actorPara);
        criticMainNet.Save(criticPara);
        return;
    }

    void DDPG::Load(const std::string& actorPara, const std::string& criticPara)
    {
        actorMainNet.Load(actorPara);
        actorMainNet.SoftUpdateTo(actorTargetNet, alpha);
        criticMainNet.Load(criticPara);
        criticMainNet.SoftUpdateTo(criticTargetNet, alpha);
        return;
    }

}
