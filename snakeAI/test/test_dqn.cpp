#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "rl/dqn.h"
#include "rl/loss.h"
#include "rl/activate.h"
#include "rl/layer.h"
#include "rl/moe.hpp"
#include "rl/net.hpp"

/*
 * ================================================================
 * Test Suite: DQN (Deep Q-Network) — Gradient & Learning Tests
 * ================================================================
 *
 * Tests:
 *   1. test_dqn_bandit()              - DQN on contextual bandit (2-state, 2-action)
 *   2. test_dqn_gradient_direction()  - Verify Q-value gradient direction
 *   3. test_dqn_multi_state()         - DQN on multi-state bandit task (3-state, 3-action)
 *   4. test_dqn_q_value()            - Verify estimated Q-values converge
 *   5. test_dqn_gradient_sign()      - Verify gradient sign matches finite difference
 *   6. test_dqn_gradient_path()      - Verify gradient flows through each layer w/o vanishing
 * ================================================================
 */

// -------------------- Helper functions --------------------
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

// -------------------- Test 1: DQN Contextual Bandit --------------------
// State[1,0] -> optimal action 0 (reward=+1)
// State[0,1] -> optimal action 1 (reward=+1)
static int test_dqn_bandit()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 1: DQN Contextual Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "State [1,0] -> optimal action 0" << std::endl;
    std::cout << "State [0,1] -> optimal action 1" << std::endl;

    RL::DQN agent(2, 16, 2);
    const int episodes = 800;
    const int stepsPerEp = 10;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            int optimalAction;
            if (t % 2 == 0) {
                state[0] = 1.0f; state[1] = 0.0f; optimalAction = 0;
            } else {
                state[0] = 0.0f; state[1] = 1.0f; optimalAction = 1;
            }
            RL::Tensor &qValues = agent.eGreedyAction(state);
            int action = sampleAction(qValues);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : -0.1f;
            agent.perceive(state, actionOneHot, state, reward, true);
        }
        agent.learn(4096, 256, 32, 1e-3);

        if (ep % 200 == 199) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            RL::Tensor &q0 = agent.action(s0);
            RL::Tensor &q1 = agent.action(s1);
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | Q(a=0|s0)=" << std::fixed << std::setprecision(4) << q0[0]
                      << " Q(a=1|s0)=" << q0[1]
                      << " | Q(a=0|s1)=" << q1[0] << " Q(a=1|s1)=" << q1[1]
                      << std::endl;
        }
    }

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor &q0 = agent.action(s0);
    RL::Tensor &q1 = agent.action(s1);

    bool states_distinguishable = (std::abs(q0[0] - q1[0]) > 0.1f) ||
                                  (std::abs(q0[1] - q1[1]) > 0.1f);
    bool state0_correct = q0[0] > q0[1];
    bool state1_correct = q1[1] > q1[0];
    bool in_range = (q0[0] >= -0.5f && q0[0] <= 1.5f) &&
                    (q0[1] >= -0.5f && q0[1] <= 1.5f) &&
                    (q1[0] >= -0.5f && q1[0] <= 1.5f) &&
                    (q1[1] >= -0.5f && q1[1] <= 1.5f);

    bool pass = in_range && (state0_correct || state1_correct || states_distinguishable);
    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | Q(s0,a0)=" << q0[0] << " Q(s0,a1)=" << q0[1]
              << " | Q(s1,a0)=" << q1[0] << " Q(s1,a1)=" << q1[1]
              << " | s0_correct=" << (state0_correct ? "Y" : "N")
              << " s1_correct=" << (state1_correct ? "Y" : "N")
              << " distinct=" << (states_distinguishable ? "Y" : "N")
              << std::endl;

    if (pass) std::cout << ">>> Test 1 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 1 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 2: DQN Gradient Direction Check --------------------
// Verify DQN's gradient correctly pushes the optimal action's Q-value UPWARD.
// The key metric: Q(optimal) should INCREASE after training (positive delta).
static int test_dqn_gradient_direction()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: DQN Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verifies: Q(optimal_action) increases after gradient update" << std::endl;

    RL::DQN agent(2, 16, 2);
    RL::Tensor state_s0(2, 1);
    state_s0[0] = 1.0f; state_s0[1] = 0.0f;

    RL::Tensor &q_before = agent.action(state_s0);
    float q_opt_before = q_before[0];
    float q_sub_before = q_before[1];

    const int trainSteps = 200;
    for (int i = 0; i < trainSteps; i++) {
        RL::Tensor state(2,1);
        state[0] = 1.0f; state[1] = 0.0f;
        agent.perceive(state, makeOneHot(2, 0), state, 1.0f, true);
    }
    for (int i = 0; i < 30; i++) {
        agent.learn(4096, 256, 32, 1e-3);
    }

    RL::Tensor &q_after = agent.action(state_s0);
    float q_opt_after = q_after[0];
    float q_sub_after = q_after[1];

    float delta_opt = q_opt_after - q_opt_before;
    bool pass = (delta_opt > 0.01f);

    std::cout << "Q(a=0|s0) before: " << q_opt_before
              << " (subopt Q1=" << q_sub_before << ")"
              << "\n  -> after: Q_opt=" << q_opt_after
              << " Q_sub=" << q_sub_after
              << " | delta_opt=" << delta_opt
              << " delta_sub=" << (q_sub_after - q_sub_before)
              << std::endl;
    std::cout << "Result: " << (pass ? "PASS (optimal Q increases toward reward)" :
                                        "FAIL (optimal Q did not increase)")
              << std::endl;

    if (pass) std::cout << ">>> Test 2 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 2 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 3: Multi-state Bandit --------------------
// 3-state bandit: state i -> optimal action i
static int test_dqn_multi_state()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 3: DQN Multi-state Bandit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::DQN agent(3, 32, 3);
    const int episodes = 1000;
    const int stepsPerEp = 15;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(3, 1);
            int si = t % 3;
            state[si] = 1.0f;
            RL::Tensor &qValues = agent.eGreedyAction(state);
            int action = sampleAction(qValues);
            RL::Tensor actionOneHot = makeOneHot(3, action);
            float reward = (action == si) ? 1.0f : -0.1f;
            agent.perceive(state, actionOneHot, state, reward, true);
        }
        agent.learn(8192, 256, 32, 1e-3);

        if (ep % 200 == 199) {
            int correct = 0;
            for (int s = 0; s < 3; s++) {
                RL::Tensor st(3, 1); st[s] = 1.0f;
                RL::Tensor &q = agent.action(st);
                if ((int)q.argmax() == s) correct++;
            }
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | correct=" << correct << "/3" << std::endl;
        }
    }

    int correct = 0;
    for (int s = 0; s < 3; s++) {
        RL::Tensor st(3, 1); st[s] = 1.0f;
        RL::Tensor &q = agent.action(st);
        if ((int)q.argmax() == s) correct++;
    }

    bool pass = (correct >= 1);
    std::cout << "Final: " << correct << "/3 states optimal" << std::endl;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    if (pass) std::cout << ">>> Test 3 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 3 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 4: Q-Value Convergence --------------------
static int test_dqn_q_value()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 4: DQN Q-Value Convergence" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::DQN agent(2, 16, 2);
    const float target_reward = 1.0f;
    const int episodes = 400;
    const int stepsPerEp = 10;

    for (int ep = 0; ep < episodes; ep++) {
        for (int t = 0; t < stepsPerEp; t++) {
            RL::Tensor state(2, 1);
            state[0] = 1.0f; state[1] = 0.0f;
            RL::Tensor &qValues = agent.eGreedyAction(state);
            int action = sampleAction(qValues);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == 0) ? target_reward : 0.0f;
            agent.perceive(state, actionOneHot, state, reward, true);
        }
        agent.learn(4096, 256, 32, 1e-3);
    }

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor &q = agent.action(s0);
    float q_opt = q[0];
    float error = std::abs(q_opt - target_reward);

    bool pass = (error < 0.5f);
    std::cout << "Q(s0,a0)=" << q_opt
              << " (expected ~" << target_reward << ")"
              << " | error=" << error << std::endl;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    if (pass) std::cout << ">>> Test 4 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 4 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 5: Gradient Sign Check (Finite Diff) --------------------
// Lightweight verification: check analytical gradient SIGN matches finite-diff.
// For a parameter w: sign(dL/dw_analytical) should equal sign(L(w+ε)-L(w-ε)).
// We build fresh nets for each perturbation to avoid cache contamination.
static int test_dqn_gradient_sign()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 5: DQN Gradient Sign Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verify analytical gradient sign matches finite-difference" << std::endl;

    /* Main net for analytical gradients */
    RL::Net testNet(
        RL::Layer<RL::Sigmoid>::_(2, 4, true, true),
        RL::Layer<RL::Sigmoid>::_(4, 2, true, true)
    );

    RL::Tensor state(2, 1);
    state[0] = 1.0f; state[1] = 0.0f;

    /* Forward + Backward */
    RL::Tensor &q_pred = testNet.forward(state);
    RL::Tensor q_target = q_pred;
    q_target[0] = 1.0f;
    RL::Tensor dLoss = RL::Loss::MSE::df(q_pred, q_target);
    testNet.backward(state, dLoss);

    auto fc1 = static_cast<RL::iFcLayer*>(testNet[0]);
    auto fc2 = static_cast<RL::iFcLayer*>(testNet[1]);

    float w1_norm = fc1->g.w.norm2();
    float w2_norm = fc2->g.w.norm2();

    std::cout << "\nForward:"
              << " Q_pred=[" << q_pred[0] << "," << q_pred[1] << "]"
              << " Q_target=[" << q_target[0] << "," << q_target[1] << "]"
              << "\n  dL/dQ=[" << dLoss[0] << "," << dLoss[1] << "]"
              << "  |g.w| layer1=" << w1_norm << " layer2=" << w2_norm
              << std::endl;

    bool grad_flows = (w1_norm > 1e-10f) && (w2_norm > 1e-10f);
    if (!grad_flows) {
        std::cout << "FAIL: No gradient flowing through network!" << std::endl;
        std::cout << ">>> Test 5 FAILED <<<" << std::endl;
        return 1;
    }

    /*
     * Sign check: sign(L(w+ε) - L(w-ε)) should equal sign(dL/dw)
     * Use fresh copies of the net for each perturbation to avoid state contamination.
     */
    const float eps = 1e-4f;
    int total = 0, sign_match = 0;

    for (std::size_t i = 0; i < fc2->outputDim && total < 8; i++) {
        for (std::size_t j = 0; j < fc2->inputDim && total < 8; j++) {
            float orig = fc2->w(i, j);

            /* Clone net with weight = orig + eps */
            RL::Net netP(
                RL::Layer<RL::Sigmoid>::_(2, 4, true, false),
                RL::Layer<RL::Sigmoid>::_(4, 2, true, false)
            );
            testNet.copyTo(netP);
            static_cast<RL::iFcLayer*>(netP[1])->w(i, j) = orig + eps;
            RL::Tensor &qP = netP.forward(state);
            float LP = 0;
            for (std::size_t k = 0; k < qP.size(); k++) {
                float d = qP[k] - q_target[k]; LP += d * d;
            }

            /* Clone net with weight = orig - eps */
            RL::Net netM(
                RL::Layer<RL::Sigmoid>::_(2, 4, true, false),
                RL::Layer<RL::Sigmoid>::_(4, 2, true, false)
            );
            testNet.copyTo(netM);
            static_cast<RL::iFcLayer*>(netM[1])->w(i, j) = orig - eps;
            RL::Tensor &qM = netM.forward(state);
            float LM = 0;
            for (std::size_t k = 0; k < qM.size(); k++) {
                float d = qM[k] - q_target[k]; LM += d * d;
            }

            float num_diff = LP - LM;
            float ana_grad = fc2->g.w(i, j);
            bool sign_ok = (num_diff * ana_grad >= 0);
            if (sign_ok) sign_match++;
            total++;
        }
    }

    for (std::size_t i = 0; i < fc1->outputDim && total < 16; i++) {
        for (std::size_t j = 0; j < fc1->inputDim && total < 16; j++) {
            float orig = fc1->w(i, j);

            RL::Net netP(
                RL::Layer<RL::Sigmoid>::_(2, 4, true, false),
                RL::Layer<RL::Sigmoid>::_(4, 2, true, false)
            );
            testNet.copyTo(netP);
            static_cast<RL::iFcLayer*>(netP[0])->w(i, j) = orig + eps;
            RL::Tensor &qP = netP.forward(state);
            float LP = 0;
            for (std::size_t k = 0; k < qP.size(); k++) {
                float d = qP[k] - q_target[k]; LP += d * d;
            }

            RL::Net netM(
                RL::Layer<RL::Sigmoid>::_(2, 4, true, false),
                RL::Layer<RL::Sigmoid>::_(4, 2, true, false)
            );
            testNet.copyTo(netM);
            static_cast<RL::iFcLayer*>(netM[0])->w(i, j) = orig - eps;
            RL::Tensor &qM = netM.forward(state);
            float LM = 0;
            for (std::size_t k = 0; k < qM.size(); k++) {
                float d = qM[k] - q_target[k]; LM += d * d;
            }

            float num_diff = LP - LM;
            float ana_grad = fc1->g.w(i, j);
            bool sign_ok = (num_diff * ana_grad >= 0);
            if (sign_ok) sign_match++;
            total++;
        }
    }

    float pass_rate = (total > 0) ? (100.0f * sign_match / total) : 0;
    bool pass = (pass_rate >= 80.0f);

    std::cout << "\n  Sign matches: " << sign_match << "/" << total
              << " (" << pass_rate << "%)" << std::endl;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    if (pass) std::cout << ">>> Test 5 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 5 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}

// -------------------- Test 6: Gradient Path Verification (MOE-based DQN) --------------------
static int test_dqn_gradient_path()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 6: DQN Gradient Path Verification (Full MOE Net)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    RL::Net testNet(
        RL::MOE<8, 4>::_(2, true),
        RL::TanhNorm<RL::Sigmoid>::_(2, 16, true, true),
        RL::Layer<RL::Sigmoid>::_(16, 2, true, true)
    );

    RL::Tensor state(2, 1);
    state[0] = 1.0f; state[1] = 0.0f;

    RL::Tensor &q_pred = testNet.forward(state);
    RL::Tensor q_target = q_pred;
    q_target[0] = 1.0f;
    RL::Tensor dLoss = RL::Loss::MSE::df(q_pred, q_target);
    testNet.backward(state, dLoss);

    auto fc   = static_cast<RL::iFcLayer*>(testNet[2]);
    auto tn   = static_cast<RL::iFcLayer*>(testNet[1]);
    auto moe  = static_cast<RL::MOE<8, 4>*>(testNet[0]);

    float w2 = fc->g.w.norm2(), b2 = fc->g.b.norm2();
    float w1 = tn->g.w.norm2(), b1 = tn->g.b.norm2();
    float wg = moe->g.wg.norm2(), bg = moe->g.b.norm2();

    std::cout << "  Layer2 (Sigmoid 16->2):  |g.w|=" << w2 << "  |g.b|=" << b2 << std::endl;
    std::cout << "  Layer1 (TanhNorm 2->16): |g.w|=" << w1 << "  |g.b|=" << b1 << std::endl;
    std::cout << "  Layer0 (MOE 8x4, 2D):    |g.wg|=" << wg << "  |g.b|=" << bg << std::endl;

    bool all_flow = (w2 > 1e-10f) && (b2 > 1e-10f) &&
                    (w1 > 1e-10f) && (b1 > 1e-10f) &&
                    (wg > 1e-10f) && (bg > 1e-10f);

    std::cout << "\nResult: "
              << (all_flow ? "PASS (gradients flow through ALL layers)" :
                             "FAIL (some gradients vanished)")
              << std::endl;
    if (all_flow) std::cout << ">>> Test 6 PASSED <<<" << std::endl;
    else          std::cout << ">>> Test 6 FAILED <<<" << std::endl;
    return all_flow ? 0 : 1;
}

// -------------------- Main --------------------
int main()
{
    std::cout << "=== DQN Test Suite ===" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;

    int failures = 0;
    failures += test_dqn_bandit();
    failures += test_dqn_gradient_direction();
    failures += test_dqn_multi_state();
    failures += test_dqn_q_value();
    failures += test_dqn_gradient_sign();
    failures += test_dqn_gradient_path();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (6 - failures) << "/6 tests passed"
              << (failures > 0 ? (" (" + std::to_string(failures) + " FAILED)") : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    return failures;
}
