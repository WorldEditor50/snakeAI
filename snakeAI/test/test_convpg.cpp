/*
 * test_convpg.cpp — ConvPG (convolutional policy gradient) test suite.
 *
 * ConvPG had NO test coverage at all before this file. Its policy network
 * consumes the whole 1x118x118 board, so the "states" below are two small board
 * images distinguished by where their single occupied cell sits.
 *
 * Tests:
 *   1. Context bandit     — the conv policy must map each board to its own
 *                           optimal action (verifies the spatial pathway can
 *                           see the board at all)
 *   2. Gradient direction — one update must move probability toward the
 *                           rewarded action on the same board
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "rl/convpg.h"
#include "rl/loss.h"
#include "rl/activate.h"

using RL::Tensor;

static int sampleAction(const Tensor &prob)
{
    return RL::Random::categorical(prob);
}

static Tensor makeOneHot(int dim, int k)
{
    Tensor a(dim, 1);
    a.zero();
    a[k] = 1.0f;
    return a;
}

/* A 1x118x118 board with a single occupied cell — the marker's position is the
   whole "state" as far as the network is concerned. */
static Tensor makeBoard(int marker)
{
    Tensor m(1, 118, 118);
    m(0, marker, marker) = 1.0f;
    return m;
}

// -------------------- Test 1: ConvPG Context Bandit --------------------
// boardA -> action 0 optimal, boardB -> action 1 optimal
static int test_convpg_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: ConvPG Context Bandit (full 118x118 board input)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::ConvPG agent(4, 8, 2);

    Tensor boardA = makeBoard(30);
    Tensor boardB = makeBoard(80);

    const int episodes = 200;
    const int stepsPerEp = 4;
    float pA_first = 0, pA_last = 0;
    int evalIdx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;

        for (int t = 0; t < stepsPerEp; t++) {
            int pattern = t % 2;                       /* alternate the board */
            Tensor &board = (pattern == 0) ? boardA : boardB;
            int optimal = pattern;                     /* A -> 0, B -> 1 */

            Tensor prob = agent.action(board);          /* value copy */
            int action = sampleAction(prob);
            Tensor oneHot = makeOneHot(2, action);
            float reward = (action == optimal) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(board, oneHot, reward));
        }

        agent.reinforce1(trajectory, 1e-2f);

        if (ep % 50 == 49) {
            Tensor pA = agent.action(boardA);           /* value copies */
            Tensor pB = agent.action(boardB);
            if (evalIdx == 0) { pA_first = pA[0]; }
            pA_last = pA[0];
            evalIdx++;
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|boardA)=" << std::fixed << std::setprecision(3) << pA[0]
                      << " P(a=1|boardB)=" << pB[1]
                      << std::endl;
        }
    }

    Tensor pA = agent.action(boardA);                   /* value copies */
    Tensor pB = agent.action(boardB);
    bool pass = (pA[0] > 0.6f && pB[1] > 0.6f);
    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|boardA)=" << pA[0] << " P(a=1|boardB)=" << pB[1]
              << "  [P(a=0|boardA): " << pA_first << " -> " << pA_last << "]"
              << std::endl;

    if (pass) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 1 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 2: ConvPG Gradient Direction --------------------
// One update on a single board: steps that take a=0 and are rewarded push
// P(a=0) up, steps that take a=1 and get nothing push it up as well (by
// suppressing a=1). Every step uses the SAME board so the sign cannot be blurred
// by spatial generalisation between two different boards.
static int test_convpg_gradient()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: ConvPG Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    Tensor board = makeBoard(40);

    std::vector<RL::Step> trajectory;
    trajectory.push_back(RL::Step(board, makeOneHot(2, 0), 1.0f));
    trajectory.push_back(RL::Step(board, makeOneHot(2, 1), 0.0f));
    trajectory.push_back(RL::Step(board, makeOneHot(2, 0), 1.0f));
    trajectory.push_back(RL::Step(board, makeOneHot(2, 1), 0.0f));

    RL::ConvPG agent(4, 8, 2);
    Tensor before = agent.action(board);               /* value copy */
    agent.reinforce1(trajectory, 1e-2f);
    Tensor after = agent.action(board);                /* value copy */

    bool pass = (after[0] > before[0]);
    std::cout << "P(a=0) before: " << before[0] << " -> after: " << after[0]
              << " (delta=" << (after[0] - before[0]) << ")" << std::endl;
    std::cout << "Result: " << (pass ? "PASS (gradient increases the rewarded action)"
                                     : "FAIL (gradient should increase the rewarded action)")
              << std::endl;

    if (pass) std::cout << ">>> Test 2 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 2 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

int main()
{
    std::cout << "=== ConvPG Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;

    int failures = 0;
    failures += test_convpg_bandit();
    failures += test_convpg_gradient();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (2 - failures) << "/2 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return failures;
}
