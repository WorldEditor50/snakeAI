#include "qlstm.h"
#include "lstm.h"
#include "layer.h"
#include "loss.h"
#include <limits>

RL::QLSTM::QLSTM(std::size_t stateDim_, std::size_t hiddenDim_, std::size_t actionDim_)
    :stateDim(stateDim_), actionDim(actionDim_), gamma(0.99), exploringRate(1)
{
    h = Tensor(hiddenDim_, 1);
    c = Tensor(hiddenDim_, 1);
    lstm = LSTM::_(stateDim_, hiddenDim_, hiddenDim_, true);
    lstmTarget = LSTM::_(stateDim_, hiddenDim_, hiddenDim_, false);
    QMainNet = Net(lstm,
                   TanhNorm<Sigmoid>::_(hiddenDim_, hiddenDim_, true, true),
                   Layer<Sigmoid>::_(hiddenDim_, actionDim_, true, true));
    QTargetNet = Net(lstmTarget,
                     TanhNorm<Sigmoid>::_(hiddenDim_, hiddenDim_, true, false),
                     Layer<Sigmoid>::_(hiddenDim_, actionDim_, true, false));
    QMainNet.copyTo(QTargetNet);
}

void RL::QLSTM::perceive(Tensor& state,
        Tensor& action,
        Tensor& nextState,
        float reward,
        bool done)
{
    memories.push_back(Transition(state, action, nextState, reward, done));
    seqEnds.push_back(done);
    if (done) {
        currentSeqId++;
    }
    return;
}

RL::Tensor& RL::QLSTM::eGreedyAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state, true);
    return eGreedy(out, exploringRate, false);
}

RL::Tensor& RL::QLSTM::noiseAction(const Tensor &state)
{
    Tensor& out = QMainNet.forward(state, true);
    return noise(out, exploringRate);
}

RL::Tensor &RL::QLSTM::action(const Tensor &state)
{
    lstm->h = h;
    lstm->c = c;
    return QMainNet.forward(state, true);
}

void RL::QLSTM::reset()
{
    lstm->reset();
    return;
}

void RL::QLSTM::experienceReplaySeq(int startIdx, int seqLen)
{
    /*
     * Process a consecutive sequence through LSTM
     * for proper BPTT (Backpropagation Through Time)
     */
    for (int t = 0; t < seqLen; t++) {
        int idx = startIdx + t;
        const Transition &x = memories[idx];

        /* Forward through QMainNet (train mode: cache states for LSTM) */
        Tensor out = QMainNet.forward(x.state, false);
        int i = x.action.argmax();

        /* Compute Q-target */
        Tensor qTarget = out;
        if (x.done) {
            qTarget[i] = x.reward;
        } else {
            /* Double DQN: action from QMainNet, value from QTargetNet */
            int k_opt = QMainNet.forward(x.nextState, true).argmax();
            Tensor &v = QTargetNet.forward(x.nextState, true);
            qTarget[i] = x.reward + gamma * v[k_opt];
        }

        /* Accumulate gradient for Q-regression */
        QMainNet.backward(x.state, Loss::MSE::df(out, qTarget));
    }
    return;
}

void RL::QLSTM::learn(std::size_t maxMemorySize,
                    std::size_t replaceTargetIter,
                    std::size_t batchSize,
                    float learningRate)
{
    h = lstm->h;
    c = lstm->c;

    if (memories.size() < batchSize) {
        return;
    }

    /* Soft update target network */
    float tau = 5e-3;
    QMainNet.softUpdateTo(QTargetNet, tau);

    /* ================================================
     * Sequence-based Experience Replay
     * Sample consecutive segments for LSTM BPTT
     * ================================================ */
    int seqLen = 8;

    /* Find sequences by episode boundaries */
    std::vector<std::pair<int, int>> completeSeqs;
    int seqStart = 0;
    for (std::size_t i = 0; i < seqEnds.size(); i++) {
        if (seqEnds[i]) {
            int len = i - seqStart + 1;
            if (len >= 2) {
                completeSeqs.push_back({seqStart, len});
            }
            seqStart = i + 1;
        }
    }
    if (seqStart < (int)memories.size() - 1) {
        int len = memories.size() - seqStart;
        if (len >= 2) {
            completeSeqs.push_back({seqStart, len});
        }
    }

    /* Train on sequences */
    if (!completeSeqs.empty()) {
        for (std::size_t s = 0; s < batchSize / seqLen + 1; s++) {
            std::uniform_int_distribution<int> seqPick(0, completeSeqs.size() - 1);
            int pick = seqPick(Random::engine);
            int start = completeSeqs[pick].first;
            int len = completeSeqs[pick].second;
            int subLen = std::min(len, seqLen);
            if (subLen >= 2) {
                std::uniform_int_distribution<int> subStart(0, len - subLen);
                int offset = subStart(Random::engine);
                lstm->reset();
                lstmTarget->reset();
                experienceReplaySeq(start + offset, subLen);
            }
        }
    } else {
        /* Fallback: sample random segments */
        std::uniform_int_distribution<int> uniform(0, memories.size() - seqLen);
        for (std::size_t s = 0; s < batchSize / seqLen + 1; s++) {
            int start = uniform(Random::engine);
            lstm->reset();
            lstmTarget->reset();
            experienceReplaySeq(start, seqLen);
        }
    }

    /* Apply optimizer */
    QMainNet.RMSProp(learningRate * 0.5, 0.9, 0);

    /* Reduce memory */
    if (memories.size() > maxMemorySize) {
        std::size_t k = memories.size() / 4;
        for (std::size_t i = 0; i < k; i++) {
            memories.pop_front();
            seqEnds.pop_front();
        }
    }

    /* Update step */
    exploringRate *= 0.99999;
    exploringRate = exploringRate < 0.1 ? 0.1 : exploringRate;
    learningSteps++;
    return;
}
