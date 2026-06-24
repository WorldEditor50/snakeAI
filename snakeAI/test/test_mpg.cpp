#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "../rl/mpg.h"
#include "../rl/loss.h"
#include "../rl/activate.h"

/*
 * ================================================================
 * Test Suite: MPG (Mamba Policy Gradient)
 * ================================================================
 *
 * IMPORTANT: Net::forward() returns an internal reference.  Calling
 * action() twice without copying gives both variables the same value
 * (the second call's output).  All tests MUST use value copies:
 *   Tensor p0 = agent.action(s0);   // value copy, OK
 *   Tensor &p0 = agent.action(s0);  // BUG: aliased, will be overwritten
 *
 * Tests:
 *   1. test_mpg_bandit()         - MPG on contextual bandit (memoryless)
 *   2. test_mpg_gradient()       - MPG numerical gradient direction check
 *   3. test_mpg_sequence()       - MPG with Mamba temporal credit assignment
 *   4. test_mpg_no_memory()      - MPG on memoryless task (Mamba not needed)
 * ================================================================
 */

// -------------------- Helpers --------------------
static int sampleAction(const RL::Tensor &prob)
{
    return RL::Random::categorical(prob);
}

static RL::Tensor makeOneHot(int dim, int k)
{
    RL::Tensor a(dim, 1);
    a.zero();
    a[k] = 1.0f;
    return a;
}

// -------------------- Test 1: MPG Contextual Bandit --------------------
// State[1,0] -> optimal action 0 (reward=+1)
// State[0,1] -> optimal action 1 (reward=+1)
static int test_mpg_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: MPG Contextual Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "State [1,0] -> optimal action 0" << std::endl;
    std::cout << "State [0,1] -> optimal action 1" << std::endl;

    RL::MPG agent(2, 16, 2);

    const int episodes = 1000;
    const int stepsPerEp = 10;

    float p0_history[5], p1_history[5];
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;
        float epReward = 0;
        agent.resetState();

        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor prob = agent.action(state); // COPY
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
            epReward += reward;
        }

        agent.reinforce(trajectory, 1e-3f);

        // Evaluate every 200 episodes
        if (ep % 200 == 199) {
            agent.resetState();
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor p0 = agent.action(s0);  // COPY
            RL::Tensor p1 = agent.action(s1);  // COPY
            p0_history[eval_idx] = p0[0];
            p1_history[eval_idx] = p1[1];
            eval_idx++;

            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | avgR=" << std::fixed << std::setprecision(2) << epReward/stepsPerEp
                      << " | P(a=0|s0)=" << p0[0]
                      << " P(a=1|s1)=" << p1[1]
                      << std::endl;
        }
    }

    /* Final evaluation */
    agent.resetState();
    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor p0 = agent.action(s0);  // COPY
    RL::Tensor p1 = agent.action(s1);  // COPY

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    bool trend_ok = (p0_history[0] < p0_history[eval_idx-1] - 0.05f)
                 || (p1_history[0] < p1_history[eval_idx-1] - 0.05f);

    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << std::endl;
    if (!trend_ok) {
        std::cout << "WARNING: Optimal action probability did not increase over training" << std::endl;
    }

    if (pass) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 1 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}


// -------------------- Test 2: MPG Gradient Direction Check --------------------
static int test_mpg_gradient()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: MPG Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verifies that REINFORCE update increases P(optimal action)" << std::endl;

    RL::Tensor state_s0(2, 1);
    state_s0[0] = 1.0f; state_s0[1] = 0.0f;
    RL::Tensor state_s1(2, 1);
    state_s1[0] = 0.0f; state_s1[1] = 1.0f;

    /* Test 2a: optimal action with high reward -> P(opt) should increase */
    RL::MPG agent1(2, 8, 2);
    RL::Tensor prob_before = agent1.action(state_s0);  // COPY
    float p_opt_before = prob_before[0];

    std::vector<RL::Step> trajectory;
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 0), 1.0f));

    agent1.reinforce(trajectory, 5e-3f);

    RL::Tensor prob_after = agent1.action(state_s0);  // COPY
    float p_opt_after = prob_after[0];
    bool pass = (p_opt_after > p_opt_before);
    std::cout << "Test 2a: P(a=0|s0) before: " << p_opt_before
              << " -> after: " << p_opt_after
              << " (delta=" << (p_opt_after - p_opt_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass ? "PASS (gradient increases optimal prob)" :
                                        "FAIL (gradient should increase optimal prob)")
              << std::endl;

    /* Test 2b: suboptimal action with zero reward -> P(subopt) should decrease */
    RL::MPG agent2(2, 8, 2);
    RL::Tensor p2_before = agent2.action(state_s0);  // COPY
    float p2_opt_before = p2_before[0];

    std::vector<RL::Step> trajectory2;
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 1), 0.0f));

    agent2.reinforce(trajectory2, 5e-3f);

    RL::Tensor p2_after = agent2.action(state_s0);  // COPY
    float p2_opt_after = p2_after[0];

    bool pass2 = (p2_opt_after < p2_opt_before);
    std::cout << "\nTest 2b: suboptimal action a=1 with low reward" << std::endl;
    std::cout << "P(a=0|s0) before: " << p2_opt_before
              << " -> after: " << p2_opt_after
              << " (delta=" << (p2_opt_after - p2_opt_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass2 ? "PASS (gradient decreases optimal prob)" :
                                        "FAIL (gradient should decrease optimal prob)")
              << std::endl;

    bool overall = pass && pass2;
    if (overall) std::cout << "\n>>> Test 2 PASSED <<<" << std::endl;
    else         std::cout << "\n>>> Test 2 FAILED <<<" << std::endl;
    return overall ? 0 : 1;
}


// -------------------- Test 3: MPG with Mamba Sequence Learning --------------------
// 2-step sequence with temporal dependency on Mamba hidden state.
// The Mamba must encode step 0's context in hidden state to inform step 1.
static int test_mpg_sequence()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: MPG with Mamba Temporal Sequence" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "2-step sequence with temporal dependency via Mamba hidden state" << std::endl;
    std::cout << "Pattern A: s0=[1,0,0] -> s1 optimal = 0" << std::endl;
    std::cout << "Pattern B: s0=[0,0,1] -> s1 optimal = 1" << std::endl;

    RL::MPG agent(3, 8, 2);

    const int episodes = 2000;
    const int evalInterval = 400;
    const int PATTERN_A = 0;
    const int PATTERN_B = 1;

    float eval_correct_rate[5];
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        int pattern = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

        /* Step 0 */
        RL::Tensor s0(3, 1);
        s0[pattern == PATTERN_A ? 0 : 2] = 1.0f;
        RL::Tensor p0 = agent.action(s0);  // COPY
        int a0 = sampleAction(p0);
        RL::Tensor a0OneHot = makeOneHot(2, a0);

        /* Step 1 */
        RL::Tensor s1(3, 1);
        s1[1] = 1.0f;
        RL::Tensor p1 = agent.action(s1);  // COPY
        int a1 = sampleAction(p1);
        RL::Tensor a1OneHot = makeOneHot(2, a1);

        int optimalA1 = (pattern == PATTERN_A) ? 0 : 1;
        float reward = (a1 == optimalA1) ? 1.0f : 0.0f;

        std::vector<RL::Step> trajectory;
        trajectory.push_back(RL::Step(s0, a0OneHot, 0.0f));
        trajectory.push_back(RL::Step(s1, a1OneHot, reward));

        agent.reinforce(trajectory, 5e-4f);

        /* Evaluate */
        if (ep % evalInterval == evalInterval - 1) {
            int correct = 0;
            const int evalEpisodes = 50;
            for (int e = 0; e < evalEpisodes; e++) {
                int p = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

                RL::Tensor es0(3, 1);
                es0[p == PATTERN_A ? 0 : 2] = 1.0f;
                agent.action(es0);  // propagate Mamba state, discard output

                RL::Tensor es1(3, 1);
                es1[1] = 1.0f;
                RL::Tensor ep1 = agent.action(es1);  // COPY
                int ea1 = ep1.argmax();

                int eOptimal = (p == PATTERN_A) ? 0 : 1;
                if (ea1 == eOptimal) correct++;
            }
            float rate = float(correct) / float(evalEpisodes);
            eval_correct_rate[eval_idx++] = rate;
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | s1 correct=" << correct << "/" << evalEpisodes
                      << " (" << std::fixed << std::setprecision(1) << rate*100 << "%)"
                      << std::endl;
        }
    }

    bool pass = (eval_correct_rate[eval_idx-1] > 0.6f);
    bool trend = (eval_idx >= 2 && eval_correct_rate[eval_idx-1] > eval_correct_rate[0]);

    std::cout << "\nSummary: final=" << std::fixed << std::setprecision(1)
              << (eval_correct_rate[eval_idx-1]*100) << "%"
              << " (baseline=50%)"
              << std::endl;

    if (pass && trend) std::cout << ">>> Test 3 PASSED <<<" << std::endl;
    else               std::cout << ">>> Test 3 FAILED <<<" << std::endl;
    return (pass && trend) ? 0 : 1;
}


// -------------------- Test 4: MPG on Memoryless Task --------------------
static int test_mpg_no_memory()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: MPG on Memoryless Task" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Same contextual bandit as Test 1 with MPG (Mamba)." << std::endl;

    RL::MPG agent(2, 8, 2);

    const int episodes = 1000;
    int eval_idx = 0;
    float p0_vals[5], p1_vals[5];

    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;
        agent.resetState();

        for (int t = 0; t < 10; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor prob = agent.action(state);  // COPY
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
        }

        agent.reinforce(trajectory, 5e-4f);

        if (ep % 200 == 199) {
            agent.resetState();
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor p0 = agent.action(s0);  // COPY
            RL::Tensor p1 = agent.action(s1);  // COPY
            p0_vals[eval_idx] = p0[0];
            p1_vals[eval_idx] = p1[1];
            eval_idx++;

            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|s0)=" << p0[0]
                      << " P(a=1|s1)=" << p1[1]
                      << std::endl;
        }
    }

    agent.resetState();
    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor p0 = agent.action(s0);  // COPY
    RL::Tensor p1 = agent.action(s1);  // COPY

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << std::endl;

    if (pass) std::cout << ">>> Test 4 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 4 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}


// -------------------- Main --------------------
int main()
{
    std::cout << "=== Mamba Policy Gradient (MPG) Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Framework: SimpleRL (MPG = Policy Gradient + Mamba SSM)" << std::endl;

    int failures = 0;

    failures += test_mpg_bandit();
    failures += test_mpg_gradient();
    failures += test_mpg_sequence();
    failures += test_mpg_no_memory();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (4 - failures) << "/4 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return failures;
}
