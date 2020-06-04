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
        this->ActorMainNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 1, ACTIVATE_SIGMOID, LOSS_KL);
        this->ActorTargetNet.CreateNet(stateDim, hiddenDim, hiddenLayerNum, actionDim, 0, ACTIVATE_SIGMOID, LOSS_KL);
        this->ActorMainNet.SoftUpdateTo(ActorTargetNet, alpha);
        /* critic: Q(S, A, α, β) = V(S, α) + A(S, A, β) */
        this->CriticMainNet.CreateNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 1);
        this->CriticTargetNet.CreateNet(stateDim + actionDim, hiddenDim, hiddenLayerNum, actionDim, 0);
        this->CriticMainNet.SoftUpdateTo(CriticTargetNet, alpha);
        return;
    }

    void DDPG::Perceive(std::vector<double>& state,
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

    void DDPG::SetSA(std::vector<double> &state, std::vector<double> &Action)
    {
        for (int i = 0; i < stateDim; i++) {
            sa[i] = state[i];
        }
        for (int i = stateDim; i < stateDim + actionDim; i++) {
            sa[i] = Action[i];
        }
        return;
    }

    int DDPG::NoiseAction(std::vector<double> &state)
    {
        int index = 0;
        ActorMainNet.FeedForward(state);
        std::vector<double>& Action = ActorMainNet.GetOutput();
        for (int i = 0; i < actionDim; i++) {
            Action[i] += double(rand() % 100 - rand() % 100) / 1000;
        }
        index = MaxQ(Action);
        return index;
    }

    int DDPG::RandomAction()
    {
        return rand() % actionDim;
    }

    int DDPG::GreedyAction(std::vector<double> &state)
    {
        if (state.size() != stateDim) {
            return 0;
        }
        double p = double(rand() % 10000) / 10000;
        int index = 0;
        if (p < exploringRate) {
            index = rand() % actionDim;
        } else {
            index = Action(state);
        }
        return index;
    }

    int DDPG::Action(std::vector<double> &state)
    {
        int index = 0;
        ActorMainNet.FeedForward(state);
        std::vector<double>& Action = ActorMainNet.GetOutput();
        index = MaxQ(Action);
        return index;
    }

    int DDPG::MaxQ(std::vector<double>& q_value)
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

    void DDPG::ExperienceReplay(Transition& x)
    {
        std::vector<double> cTarget(actionDim);
        std::vector<double>& aMainOutput = ActorMainNet.GetOutput();
        std::vector<double>& aTargetOutput = ActorTargetNet.GetOutput();
        std::vector<double>& cTargetOutput = CriticTargetNet.GetOutput();
        std::vector<double>& cMainOutput = CriticMainNet.GetOutput();
        /* estimate Action value */
        int i = int(x.action);
        ActorMainNet.FeedForward(x.state);
        SetSA(x.state, aMainOutput);
        CriticMainNet.FeedForward(sa);
        cTarget = cMainOutput;
        if (x.done == true) {
            cTarget[i] = x.reward;
        } else {
            ActorTargetNet.FeedForward(x.nextState);
            SetSA(x.nextState, aTargetOutput);
            CriticTargetNet.FeedForward(sa);
            CriticMainNet.FeedForward(sa);
            int k = MaxQ(cMainOutput);
            cTarget[i] = x.reward + gamma * cTargetOutput[k];
        }
        /* update ActorMainNet */
        std::vector<double> delta(actionDim);
        ActorMainNet.FeedForward(x.state);
        ActorTargetNet.FeedForward(x.state);
        SetSA(x.state, aMainOutput);
        CriticMainNet.FeedForward(sa);
        ActorMainNet.Gradient(x.state, cMainOutput);
        /* update CriticMainNet */
        CriticMainNet.Gradient(sa, cTarget);
        return;
    }

    void DDPG::Learn(int optType, double actorLearningRate, double criticLearningRate)
    {
        if (memories.size() < batchSize) {
            return;
        }
        if (learningSteps % replaceTargetIter == 0) {
            std::cout<<"update target net"<<std::endl;
            /* update tagetNet */
            ActorMainNet.SoftUpdateTo(ActorTargetNet, alpha);
            CriticMainNet.SoftUpdateTo(CriticTargetNet, alpha);
            learningSteps = 0;
        }
        /* experience replay */
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % memories.size();
            ExperienceReplay(memories[k]);
        }
        ActorMainNet.Optimize(optType, actorLearningRate);
        CriticMainNet.Optimize(optType, criticLearningRate);
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
        ActorMainNet.Save(actorPara);
        CriticMainNet.Save(criticPara);
        return;
    }

    void DDPG::Load(const std::string& actorPara, const std::string& criticPara)
    {
        ActorMainNet.Load(actorPara);
        ActorMainNet.SoftUpdateTo(ActorTargetNet, alpha);
        CriticMainNet.Load(criticPara);
        CriticMainNet.SoftUpdateTo(CriticTargetNet, alpha);
        return;
    }

}
