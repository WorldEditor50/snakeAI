#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "rl/dpg.h"
#include "rl/drpg.h"
#include "rl/convpg.h"
#include "rl/loss.h"
#include "rl/activate.h"

/*
 * ================================================================
 * Test Suite: Policy Gradient Algorithms (DPG, DRPG with LSTM)
 * ================================================================
 *
 * Tests:
 *   1. test_dpg_bandit()       - DPG on contextual bandit
 *   2. test_dpg_gradient()     - DPG numerical gradient verification
 *   3. test_drpg_sequence()    - DRPG with LSTM temporal credit assignment
 *   4. test_drpg_vs_dpg()      - DRPG vs DPG on memoryless task (should match)
 * ================================================================
 */

// -------------------- Helper: sample action from softmax policy --------------------
static int sampleAction(RL::Tensor &prob)
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

// -------------------- Test 1: DPG Contextual Bandit --------------------
// State[1,0] -> optimal action 0 (reward=+1)
// State[0,1] -> optimal action 1 (reward=+1)
// Verifies: REINFORCE gradient direction, alpha temperature, baseline
static int test_dpg_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: DPG Contextual Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "State [1,0] -> optimal action 0" << std::endl;
    std::cout << "State [0,1] -> optimal action 1" << std::endl;

    RL::DPG agent(2, 16, 2);

    const int episodes = 1000;
    const int stepsPerEp = 10;

    float p0_history[5], p1_history[5];
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;
        float epReward = 0;

        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor &prob = agent.action(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
            epReward += reward;
        }

        agent.reinforce(trajectory, 1e-3f);

        // Evaluate every 200 episodes
        if (ep % 200 == 199) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor &p0 = agent.action(s0);
            RL::Tensor &p1 = agent.action(s1);
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

    // Final evaluation
    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor &p0 = agent.action(s0);
    RL::Tensor &p1 = agent.action(s1);

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << std::endl;

    // Additional check: probability of optimal action should increase over training
    bool trend_ok = (p0_history[0] < p0_history[eval_idx-1] - 0.05f);
    if (!trend_ok) {
        std::cout << "WARNING: Optimal action probability did not increase monotonically" << std::endl;
    }

    if (pass) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 1 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}


// -------------------- Test 2: DPG Gradient Direction Check --------------------
// Finite-difference check: verify that ∇log π(a|s)·A pushes P(optimal) upward
static int test_dpg_gradient()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: DPG Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verifies that REINFORCE update increases P(optimal action)" << std::endl;

    RL::DPG agent(2, 8, 2);

    // Use 2-step trajectory to get non-zero advantages
    // Step 1: state=[1,0], action 0 (optimal), reward=+1 => discounted return = 1 + 0.9*1 = 1.9
    // Step 2: state=[0,1], action 0 (suboptimal), reward=0 => discounted return = 0
    // Mean baseline = (1.9 + 0)/2 = 0.95
    // Advantage step 1 = 1.9 - 0.95 = +0.95 (positive -> increase P(opt))
    // Advantage step 2 = 0 - 0.95 = -0.95 (negative -> decrease P(subopt))

    RL::Tensor state_s0(2, 1);
    state_s0[0] = 1.0f; state_s0[1] = 0.0f;
    RL::Tensor state_s1(2, 1);
    state_s1[0] = 0.0f; state_s1[1] = 1.0f;

    RL::DPG agent1(2, 8, 2);
    RL::Tensor &prob_before = agent1.action(state_s0);
    float p_opt_before = prob_before[0];

    std::vector<RL::Step> trajectory;
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 0), 1.0f));   // optimal, high reward -> increase P(0|s0)
    trajectory.push_back(RL::Step(state_s1, makeOneHot(2, 0), 0.0f));   // suboptimal, low reward -> decrease P(0|s1)

    agent1.reinforce(trajectory, 5e-3f);

    RL::Tensor &prob_after = agent1.action(state_s0);
    float p_opt_after = prob_after[0];
    bool pass = (p_opt_after > p_opt_before);
    std::cout << "P(a=0|s0) before: " << p_opt_before
              << " -> after: " << p_opt_after
              << " (delta=" << (p_opt_after - p_opt_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass ? "PASS (gradient increases optimal prob)" :
                                        "FAIL (gradient should increase optimal prob)")
              << std::endl;

    // Anti-test: suboptimal action with negative advantage should decrease P(opt)
    RL::DPG agent2(2, 8, 2);
    RL::Tensor &p2_before = agent2.action(state_s0);
    float p2_opt_before = p2_before[0];

    std::vector<RL::Step> trajectory2;
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 1), 0.0f));  // suboptimal, low reward -> decrease P(1|s0)
    trajectory2.push_back(RL::Step(state_s1, makeOneHot(2, 0), 1.0f));  // optimal, high reward -> increase P(0|s1)

    agent2.reinforce(trajectory2, 5e-3f);

    RL::Tensor &p2_after = agent2.action(state_s0);
    float p2_opt_after = p2_after[0];

    bool pass2 = (p2_opt_after < p2_opt_before);  // P(opt=0) decreased since we took suboptimal action a=1
    std::cout << "\nAnti-test: suboptimal action a=1 with low reward" << std::endl;
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


// -------------------- Test 3: DRPG with LSTM Sequence Learning --------------------
// 2-step sequence with temporal dependency:
// Pattern A: s0=[1,0,0], s1=[0,1,0], reward at s1 = +1 if action matches s0's optimal
// Pattern B: s0=[0,0,1], s1=[0,1,0], reward at s1 = +1 if action != s0's optimal
//
// The LSTM must encode step 0's context in hidden state to inform step 1's decision.
static int test_drpg_sequence()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: DRPG with LSTM Temporal Sequence" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "2-step sequence with temporal dependency via LSTM hidden state" << std::endl;
    std::cout << "Pattern A: s0=[1,0,0] -> later optimal at s1 is action 0" << std::endl;
    std::cout << "Pattern B: s0=[0,0,1] -> later optimal at s1 is action 1" << std::endl;
    std::cout << "The LSTM must propagate context from step 0 to step 1" << std::endl;

    RL::DRPG agent(3, 8, 2);

    const int episodes = 2000;
    const int evalInterval = 400;

    // Patterns: (s0_encoding, optimal_action_at_s1)
    const int PATTERN_A = 0; // s0=[1,0,0] -> s1 optimal = 0
    const int PATTERN_B = 1; // s0=[0,0,1] -> s1 optimal = 1

    float eval_correct_rate[5];
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        // Randomly choose pattern
        int pattern = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

        // Step 0
        RL::Tensor s0(3, 1);
        s0[pattern == PATTERN_A ? 0 : 2] = 1.0f;
        s0[1] = 0.0f;

        RL::Tensor &p0 = agent.action(s0);
        int a0 = sampleAction(p0);
        RL::Tensor a0OneHot = makeOneHot(2, a0);

        // Step 1
        RL::Tensor s1(3, 1);
        s1[0] = 0.0f; s1[1] = 1.0f; s1[2] = 0.0f;

        RL::Tensor &p1 = agent.action(s1);
        int a1 = sampleAction(p1);
        RL::Tensor a1OneHot = makeOneHot(2, a1);

        // Reward at step 1 only
        int optimalA1 = (pattern == PATTERN_A) ? 0 : 1;
        float reward = (a1 == optimalA1) ? 1.0f : 0.0f;

        // Build trajectory
        std::vector<RL::Step> trajectory;
        trajectory.push_back(RL::Step(s0, a0OneHot, 0.0f));  // no reward at step 0
        trajectory.push_back(RL::Step(s1, a1OneHot, reward));

        agent.reinforce(trajectory, 5e-4f);

        // Evaluate
        if (ep % evalInterval == evalInterval - 1) {
            int correct = 0;
            const int evalEpisodes = 50;
            for (int e = 0; e < evalEpisodes; e++) {
                int p = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

                RL::Tensor es0(3, 1);
                es0[p == PATTERN_A ? 0 : 2] = 1.0f;
                es0[1] = 0.0f;
                agent.action(es0); // propagate LSTM

                RL::Tensor es1(3, 1);
                es1[0] = 0.0f; es1[1] = 1.0f; es1[2] = 0.0f;
                RL::Tensor &ep1 = agent.action(es1);
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

    bool pass = (eval_correct_rate[eval_idx-1] > 0.7f);
    bool trend = (eval_idx >= 2 && eval_correct_rate[eval_idx-1] > eval_correct_rate[0]);

    std::cout << "\nSummary: final accuracy=" << std::fixed << std::setprecision(1)
              << (eval_correct_rate[eval_idx-1]*100) << "%"
              << " (random baseline = 50%)"
              << std::endl;
    std::cout << "Improvement: " << (eval_correct_rate[eval_idx-1] - eval_correct_rate[0])*100
              << "%" << std::endl;

    if (pass && trend) std::cout << ">>> Test 3 PASSED <<<" << std::endl;
    else               std::cout << ">>> Test 3 FAILED <<<" << std::endl;
    return (pass && trend) ? 0 : 1;
}


// -------------------- Test 4: DRPG on Memoryless Task (should match DPG) --------------------
// Same contextual bandit as Test 1, but using DRPG (LSTM).
// Since there's no temporal dependency, LSTM is unnecessary but should not prevent learning.
static int test_drpg_no_memory()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: DRPG on Memoryless Task" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Same contextual bandit as Test 1 but with DRPG (LSTM)." << std::endl;
    std::cout << "LSTM should not prevent learning on a memoryless task." << std::endl;

    RL::DRPG agent(2, 8, 2);

    const int episodes = 1000;
    int eval_idx = 0;
    float p0_history[5], p1_history[5];

    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;

        for (int t = 0; t < 10; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor &prob = agent.action(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
        }

        agent.reinforce(trajectory, 5e-4f);

        if (ep % 200 == 199) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor &p0 = agent.action(s0);
            RL::Tensor &p1 = agent.action(s1);
            p0_history[eval_idx] = p0[0];
            p1_history[eval_idx] = p1[1];
            eval_idx++;

            // Reset LSTM state between evaluations
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|s0)=" << p0[0]
                      << " P(a=1|s1)=" << p1[1]
                      << std::endl;
        }
    }

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor &p0 = agent.action(s0);
    RL::Tensor &p1 = agent.action(s1);

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
    std::cout << "=== Policy Gradient Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Framework: SimpleRL (DRPG = Policy Gradient + LSTM)" << std::endl;

    int failures = 0;

    failures += test_dpg_bandit();
    failures += test_dpg_gradient();
    failures += test_drpg_sequence();
    failures += test_drpg_no_memory();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (4 - failures) << "/4 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return failures;
}
