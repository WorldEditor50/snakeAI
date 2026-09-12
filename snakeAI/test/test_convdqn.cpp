/*
 * test_convdqn.cpp — ConvDQN (convolutional deep Q-network) test suite.
 *
 * ConvDQN had NO test coverage at all before this file, which is how a Q-head
 * that cannot represent negative values survived unnoticed: the agent's only
 * visible symptom was "ConvDQN has no noticeable effect in the game".
 *
 * The task is a SPATIAL contextual bandit: the 1x118x118 board holds a single
 * marker whose position determines the optimal action. Rewards are +1 / -1, so
 * the wrong actions must end up with strictly NEGATIVE Q-values — which is what
 * exposes the activation used on the Q-head.
 *
 * Tests:
 *   1. Spatial bandit       — the conv Q-network must map each board to its own
 *                             optimal action
 *   2. Negative Q-values    — a penalised action must be representable (Q < 0)
 *   3. Exploration schedule — epsilon must actually reach its floor in a
 *                             realistic number of learn() calls
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "rl/convdqn.h"
#include "rl/activate.h"
#include "rl/util.hpp"

using RL::Tensor;

static Tensor makeOneHot(int dim, int k)
{
    Tensor a(dim, 1);
    a.zero();
    a[k] = 1.0f;
    return a;
}

/* A 1x118x118 board with a single occupied cell. */
static Tensor makeBoard(int r, int c)
{
    Tensor m(1, 118, 118);
    m(0, r, c) = 1.0f;
    return m;
}

static const int N_PATTERNS = 4;

static void makePatterns(std::vector<Tensor> &boards)
{
    boards.clear();
    boards.push_back(makeBoard(20, 20));   /* optimal action 0 */
    boards.push_back(makeBoard(20, 90));   /* optimal action 1 */
    boards.push_back(makeBoard(90, 20));   /* optimal action 2 */
    boards.push_back(makeBoard(90, 90));   /* optimal action 3 */
}

// -------------------- Test 1 + 2: Spatial bandit and negative Q-values --------------------
static int test_convdqn_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: ConvDQN Spatial Contextual Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Reward +1 for the pattern's optimal action, -1 otherwise." << std::endl;
    std::cout << "A penalised action therefore needs a NEGATIVE Q-value." << std::endl;

    RL::Random::engine.seed(20240607u);
    RL::Random::generator.seed(20240607u);
    std::srand(20240607);

    std::vector<Tensor> boards;
    makePatterns(boards);

    RL::ConvDQN agent(118 * 118, 64, N_PATTERNS);

    const int iterations = 600;
    const int batchSize = 16;

    for (int it = 0; it < iterations; it++) {
        int pattern = std::rand() % N_PATTERNS;
        const Tensor &board = boards[pattern];

        Tensor &q = agent.eGreedyAction(board);   /* epsilon-greedy (explores) */
        int action = q.argmax();
        float reward = (action == pattern) ? 1.0f : -1.0f;

        /* done = true: a bandit step, so the TD target is just the reward. */
        agent.perceive(board, makeOneHot(N_PATTERNS, action), board, reward, true);
        agent.learn(512, 256, batchSize, 1e-2f);
    }

    std::cout << "Final exploringRate: " << agent.getExploringRate() << std::endl;

    /* Greedy evaluation */
    int correct = 0;
    float worstOptimalQ = 1e30f;
    float bestPenalisedQ = -1e30f;
    for (int p = 0; p < N_PATTERNS; p++) {
        Tensor q = agent.action(boards[p]);       /* value copy */
        int greedy = q.argmax();
        if (greedy == p) correct++;

        std::cout << "pattern " << p << " Q=[" << std::fixed << std::setprecision(3);
        for (int a = 0; a < N_PATTERNS; a++) {
            std::cout << (a ? ", " : "") << q[a];
            if (a == p) {
                if (q[a] < worstOptimalQ) worstOptimalQ = q[a];
            } else {
                if (q[a] > bestPenalisedQ) bestPenalisedQ = q[a];
            }
        }
        std::cout << "]  greedy=" << greedy << (greedy == p ? " OK" : " WRONG") << std::endl;
    }

    bool pass1 = (correct == N_PATTERNS);
    std::cout << "\nResult: " << (pass1 ? "PASS" : "FAIL")
              << " | correct greedy actions: " << correct << "/" << N_PATTERNS
              << std::endl;
    if (pass1) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else       std::cout << ">>> Test 1 FAILED <<<" << std::endl;

    /* ---- Test 2: negative Q representability ---- */
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: Penalised actions must have NEGATIVE Q-values" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Worst Q over optimal actions      : " << worstOptimalQ << std::endl;
    std::cout << "Best  Q over penalised actions    : " << bestPenalisedQ << std::endl;

    bool pass2 = (worstOptimalQ > 0.0f) && (bestPenalisedQ < 0.0f);
    std::cout << "Result: " << (pass2 ? "PASS (the Q-head can represent both signs)"
                                      : "FAIL (Q-head cannot represent the penalised value)")
              << std::endl;
    if (pass2) std::cout << ">>> Test 2 PASSED <<<" << std::endl;
    else       std::cout << ">>> Test 2 FAILED <<<" << std::endl;

    return (pass1 ? 0 : 1) + (pass2 ? 0 : 1);
}

// -------------------- Test 3: Exploration schedule --------------------
static int test_convdqn_exploration_schedule()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: Exploration Schedule Reaches Its Floor" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::ConvDQN agent(118 * 118, 16, 2);

    Tensor state = makeBoard(40, 40);
    Tensor next = makeBoard(41, 40);

    const float start = agent.getExploringRate();

    /* batchSize = 1 keeps this cheap: every learn() call then costs a single
       replay, and the schedule only depends on how many times learn() runs. */
    const int batchSize = 1;
    agent.perceive(state, makeOneHot(2, 0), next, 1.0f, false);

    int needed = -1;
    const int budget = 8000;
    for (int i = 0; i < budget; i++) {
        agent.learn(4096, 256, batchSize, 1e-2f);
        if (agent.getExploringRate() <= 0.0501f) { needed = i + 1; break; }
    }

    std::cout << "exploringRate: " << start << " -> " << agent.getExploringRate()
              << " after " << (needed < 0 ? budget : needed) << " learn() calls" << std::endl;

    /* An agent that keeps exploring at ~1.0 for the whole session acts at
       random; the schedule must reach the floor inside a realistic session. */
    bool pass = (needed > 0) && (needed <= 8000);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL")
              << " (floor must be reached within 8000 learn calls)" << std::endl;

    if (pass) std::cout << ">>> Test 3 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 3 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 4: Bootstrapping (done = false) --------------------
/*
   The bandit test above only ever stores terminals (done = true), so its TD
   target is just the reward and it never exercises ConvDQN's BOOTSTRAP path.
   The game runs almost entirely on non-terminal transitions, so a broken
   bootstrap would show up in the game while the bandit test kept passing.

   Two-state chain:
     from A: action 0 -> B, reward 0, NOT done      (the only useful move)
             actions 1-3 -> A, reward -1, done
     from B: action 0 -> B, reward +1, done
             actions 1-3 -> A, reward -1, done

   The correct values are Q(B,0) = +1 and Q(A,0) = gamma * Q(B,0) = 0.99, which
   Q(A,0) can ONLY learn through bootstrapping from Q(B,0).
*/
static int test_convdqn_bootstrap()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: ConvDQN Bootstrapping From a Non-Terminal Transition" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::Random::engine.seed(777001u);
    RL::Random::generator.seed(777001u);
    std::srand(777001);

    Tensor stateA = makeBoard(20, 20);
    Tensor stateB = makeBoard(80, 80);

    RL::ConvDQN agent(118 * 118, 64, 4);

    const int iterations = 800;
    const int batchSize = 16;

    for (int it = 0; it < iterations; it++) {
        bool inA = (std::rand() % 2 == 0);

        /* Behaviour policy: mostly greedy, with decaying exploration. */
        const Tensor &s = inA ? stateA : stateB;
        Tensor &q = agent.eGreedyAction(s);
        int a = q.argmax();

        int r;
        bool done;
        const Tensor *sNext;
        if (inA) {
            if (a == 0) { r = 0;  done = false; sNext = &stateB; }
            else        { r = -1; done = true;  sNext = &stateA; }
        } else {
            if (a == 0) { r = 1;  done = true;  sNext = &stateB; }
            else        { r = -1; done = true;  sNext = &stateA; }
        }

        agent.perceive(s, makeOneHot(4, a), *sNext, float(r), done);
        agent.learn(512, 256, batchSize, 1e-2f);
    }

    Tensor qA = agent.action(stateA);   /* value copies */
    Tensor qB = agent.action(stateB);

    std::cout << "Q(A,0)=" << std::fixed << std::setprecision(3) << qA[0]
              << "  (bootstrap target gamma*Q(B,0) = 0.99)" << std::endl;
    std::cout << "Q(B,0)=" << qB[0] << "  (direct target = 1.0)" << std::endl;
    std::cout << "Q(A,1)=" << qA[1] << "  Q(B,1)=" << qB[1]
              << "  (penalised actions, target = -1.0)" << std::endl;

    bool pass = (qB[0] > 0.5f) &&          /* learned the direct reward */
                (qA[0] > 0.5f) &&          /* learned it BY BOOTSTRAPPING */
                (qA[1] < qA[0]) && (qB[1] < qB[0]);

    std::cout << "Result: " << (pass ? "PASS (the non-terminal transition propagated value)"
                                     : "FAIL (value did not propagate through the bootstrap)")
              << std::endl;

    if (pass) std::cout << ">>> Test 4 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 4 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

int main()
{
    std::cout << "=== ConvDQN Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;

    int failures = 0;
    failures += test_convdqn_bandit();
    failures += test_convdqn_bootstrap();
    failures += test_convdqn_exploration_schedule();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (5 - failures) << "/5 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return failures;
}
