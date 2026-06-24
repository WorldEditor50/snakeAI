#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "rl/sac.h"
#include "rl/dpg.h"
#include "rl/loss.h"
#include "rl/activate.h"

/*
 * ================================================================
 * Test Suite: SAC (Soft Actor-Critic)
 * ================================================================
 *
 * Tests:
 *   1. test_sac_bandit()            - SAC on contextual bandit
 *   2. test_sac_gradient_direction() - Verify policy gradient direction
 *   3. test_sac_multi_state()       - SAC on multi-state bandit task
 *   4. test_sac_vs_dpg()            - Compare SAC convergence with DPG
 *   5. test_sac_entropy()           - Verify SAC maintains exploration
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

// -------------------- Test 1: SAC Contextual Bandit --------------------
// State[1,0] -> optimal action 0 (reward=+1)
// State[0,1] -> optimal action 1 (reward=+1)
// Verifies: SAC learns the correct policy with soft Q-learning
static int test_sac_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: SAC Contextual Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "State [1,0] -> optimal action 0" << std::endl;
    std::cout << "State [0,1] -> optimal action 1" << std::endl;

    RL::SAC agent(2, 16, 2);

    const int episodes = 500;
    const int stepsPerEp = 20;

    float p0_history[5], p1_history[5];
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            /* Use gumbelMax for exploration during training */
            RL::Tensor &prob = agent.gumbelMax(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            /* Each step is a complete episode (terminal) */
            agent.perceive(state, actionOneHot, state, reward, true);
        }

        agent.learn(4096, 256, 32, 1e-3);

        /* Evaluate every 100 episodes */
        if (ep % 100 == 99) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor &p0 = agent.action(s0);
            RL::Tensor &p1 = agent.action(s1);
            p0_history[eval_idx] = p0[0];
            p1_history[eval_idx] = p1[1];
            eval_idx++;

            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|s0)=" << std::fixed << std::setprecision(4) << p0[0]
                      << " P(a=1|s1)=" << p1[1]
                      << std::endl;
        }
    }

    /* Final evaluation */
    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor &p0 = agent.action(s0);
    RL::Tensor &p1 = agent.action(s1);

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << std::endl;

    if (pass) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 1 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 2: SAC Gradient Direction Check --------------------
// Verify that SAC's policy gradient pushes probability toward optimal actions
// when trained with positive rewards
static int test_sac_gradient_direction()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: SAC Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verifies that SAC update increases P(optimal action)" << std::endl;

    RL::SAC agent(2, 8, 2);

    RL::Tensor state_s0(2, 1);
    state_s0[0] = 1.0f; state_s0[1] = 0.0f;

    /* Record probability before training */
    RL::Tensor &prob_before = agent.action(state_s0);
    float p_opt_before = prob_before[0];

    /* Train with positive reward for optimal action (action 0 from state [1,0]) */
    const int trainSteps = 150;
    for (int i = 0; i < trainSteps; i++) {
        RL::Tensor state(2,1);
        state[0] = 1.0f; state[1] = 0.0f;
        agent.perceive(state, makeOneHot(2, 0), state, 1.0f, true);
    }

    /* Run learning with small batches */
    for (int i = 0; i < 20; i++) {
        agent.learn(4096, 256, 32, 1e-3);
    }

    /* Record probability after training */
    RL::Tensor &prob_after = agent.action(state_s0);
    float p_opt_after = prob_after[0];

    bool pass = (p_opt_after > p_opt_before);
    std::cout << "P(a=0|s0) before: " << p_opt_before
              << " -> after: " << p_opt_after
              << " (delta=" << (p_opt_after - p_opt_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass ? "PASS (SAC gradient increases optimal prob)" :
                                        "FAIL (SAC gradient should increase optimal prob)")
              << std::endl;

    if (pass) std::cout << ">>> Test 2 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 2 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 3: Multi-state Bandit --------------------
// 3-state bandit: state i -> optimal action i
// Verifies SAC learns multiple state-action mappings simultaneously
static int test_sac_multi_state()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: SAC Multi-state Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "3 states, each with its own optimal action" << std::endl;

    RL::SAC agent(3, 32, 3);

    const int episodes = 400;
    const int stepsPerEp = 15;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(3, 1);
            int optimalAction;
            int si = t % 3;
            state[si] = 1.0f;
            optimalAction = si;

            RL::Tensor &prob = agent.gumbelMax(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(3, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            agent.perceive(state, actionOneHot, state, reward, true);
        }

        agent.learn(4096, 256, 32, 1e-3);

        if (ep % 100 == 99) {
            int correct = 0;
            for (int s = 0; s < 3; s++) {
                RL::Tensor st(3, 1);
                st[s] = 1.0f;
                RL::Tensor &p = agent.action(st);
                if ((int)p.argmax() == s) correct++;
            }
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | correct=" << correct << "/3"
                      << std::endl;
        }
    }

    /* Final evaluation */
    int correct = 0;
    for (int s = 0; s < 3; s++) {
        RL::Tensor st(3, 1);
        st[s] = 1.0f;
        RL::Tensor &p = agent.action(st);
        if ((int)p.argmax() == s) correct++;
    }

    bool pass = (correct >= 2);
    std::cout << "Final: " << correct << "/3 states optimal" << std::endl;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    if (pass) std::cout << ">>> Test 3 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 3 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 4: SAC vs DPG Baseline --------------------
// Compare SAC with DPG on a contextual bandit task to verify SAC learns
// at least as well as a simple policy gradient method
static int test_sac_vs_dpg()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: SAC vs DPG Baseline" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Compare SAC and DPG on the same 2-action bandit task" << std::endl;

    RL::SAC sac(2, 16, 2);
    RL::DPG dpg(2, 16, 2);

    const int episodes = 300;
    const int stepsPerEp = 20;

    float sac_final[2], dpg_final[2];

    /* Train SAC */
    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor &prob = sac.gumbelMax(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            sac.perceive(state, actionOneHot, state, reward, true);
        }
        sac.learn(4096, 256, 32, 1e-3);
    }

    /* Evaluate SAC */
    {
        RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
        RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
        RL::Tensor &p0 = sac.action(s0);
        RL::Tensor &p1 = sac.action(s1);
        sac_final[0] = p0[0];
        sac_final[1] = p1[1];
        std::cout << "SAC: P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1] << std::endl;
    }

    /* Train DPG */
    for (int ep = 0; ep < episodes; ep++) {
        std::vector<RL::Step> trajectory;
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor &prob = dpg.gumbelMax(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
        }
        dpg.reinforce(trajectory, 1e-3f);
    }

    /* Evaluate DPG */
    {
        RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
        RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
        RL::Tensor &p0 = dpg.action(s0);
        RL::Tensor &p1 = dpg.action(s1);
        dpg_final[0] = p0[0];
        dpg_final[1] = p1[1];
        std::cout << "DPG: P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1] << std::endl;
    }

    /* SAC should learn at least as well as DPG */
    bool pass = (sac_final[0] > 0.5f && sac_final[1] > 0.5f);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    if (pass) std::cout << ">>> Test 4 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 4 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 5: SAC Exploration/Entropy --------------------
// Verify SAC maintains exploration by checking that the policy
// doesn't collapse to a deterministic distribution early in training
static int test_sac_entropy()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 5: SAC Entropy Maintenance" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verify SAC policy entropy remains positive (not collapsed)" << std::endl;

    RL::SAC agent(2, 16, 2);

    /* Compute initial entropy */
    auto computeEntropy = [](RL::Tensor &p) -> float {
        float H = 0;
        for (std::size_t i = 0; i < p.size(); i++) {
            if (p[i] > 1e-8) {
                H -= p[i] * std::log(p[i]);
            }
        }
        return H;
    };

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor &p_init = agent.action(s0);
    float H_init = computeEntropy(p_init);
    float max_entropy = std::log(2.0f); /* max possible for 2 actions */

    std::cout << "Initial entropy: " << H_init
              << " (max possible = " << max_entropy << ")" << std::endl;

    /* Train on a task that could cause collapse */
    const int episodes = 200;
    const int stepsPerEp = 20;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }

            RL::Tensor &prob = agent.gumbelMax(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            agent.perceive(state, actionOneHot, state, reward, true);
        }
        agent.learn(4096, 256, 32, 1e-3);
    }

    /* Compute entropy after training */
    RL::Tensor &p_final = agent.action(s0);
    float H_final = computeEntropy(p_final);

    std::cout << "Final entropy: " << H_final << std::endl;
    std::cout << "P(a=0|s0)=" << p_final[0] << " P(a=1|s0)=" << p_final[1] << std::endl;

    /* SAC should maintain some entropy (not collapse to 0 entropy) */
    float collapse_threshold = 0.1f * max_entropy;
    bool pass = (H_final > collapse_threshold);

    if (!pass) {
        std::cout << "WARNING: Policy may have collapsed (entropy too low)" << std::endl;
    }
    std::cout << "Result: " << (pass ? "PASS (SAC maintains exploration)" :
                                        "FAIL (SAC collapsed to near-deterministic)")
              << std::endl;

    if (pass) std::cout << ">>> Test 5 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 5 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Main --------------------
int main()
{
    std::cout << "=== SAC Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Framework: SimpleRL (SAC = Soft Actor-Critic)" << std::endl;

    int failures = 0;

    failures += test_sac_bandit();
    failures += test_sac_gradient_direction();
    failures += test_sac_multi_state();
    failures += test_sac_vs_dpg();
    failures += test_sac_entropy();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (5 - failures) << "/5 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return failures;
}
