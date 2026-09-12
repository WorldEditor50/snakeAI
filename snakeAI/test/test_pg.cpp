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
            RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
            RL::Tensor p1 = agent.action(s1);   /* value copy, see above */
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
    RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
    RL::Tensor p1 = agent.action(s1);   /* value copy, see above */

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
// Single-update direction check: after one REINFORCE update, an action that
// earned a POSITIVE advantage must gain probability, and an action that earned a
// NEGATIVE advantage must lose it.
//
// IMPORTANT 鈥?every step of the trajectory must (a) act on the SAME state and
// (b) carry the SAME sign of advantage. The previous version used
// (s0, a=0, r=+1) followed by (s1, a=0, r=0): two DIFFERENT states with
// opposite-sign advantages updating the same shared weights. The change in
// P(a=0|s0) was therefore "step 0 pushes up" plus "step 1 pushes down", and the
// sign of that sum depends on how strongly this tiny network generalises between
// s0 and s1 鈥?so the check failed roughly half the time for reasons that have
// nothing to do with gradient direction.
static int test_dpg_gradient()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 2: DPG Gradient Direction Check" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Verifies that REINFORCE update increases P(optimal action)" << std::endl;

    RL::Tensor state_s0(2, 1);
    state_s0[0] = 1.0f; state_s0[1] = 0.0f;

    /*
       Every step acts on the SAME state s0. Note that a mean baseline makes the
       advantages sum to ZERO, so a multi-step trajectory always mixes positive
       and negative advantages 鈥?what has to be consistent is their EFFECT on
       P(a=0|s0) (gamma = 0.9):

         step 0: a=0, reward 1 -> G_0 = 1.810, A_0 = +0.8825 -> raise P(a=0|s0)
         step 1: a=1, reward 0 -> G_1 = 0.900, A_1 = -0.0275 -> suppress a=1, also raises P(a=0|s0)
         step 2: a=0, reward 1 -> G_2 = 1.000, A_2 = +0.0725 -> raise P(a=0|s0)
         step 3: a=1, reward 0 -> G_3 = 0.000, A_3 = -0.9275 -> suppress a=1, also raises P(a=0|s0)
         baseline = mean(G) = 0.9275

       All four gradients push the same way, and since every step is at s0 there
       is no cross-state coupling to blur the sign of the result.
    */
    std::vector<RL::Step> trajectory;
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 0), 1.0f));
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 1), 0.0f));
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 0), 1.0f));
    trajectory.push_back(RL::Step(state_s0, makeOneHot(2, 1), 0.0f));

    RL::DPG agent1(2, 8, 2);
    RL::Tensor prob_before = agent1.action(state_s0);   /* value copy */
    float p_opt_before = prob_before[0];

    agent1.reinforce(trajectory, 5e-3f);

    RL::Tensor prob_after = agent1.action(state_s0);    /* value copy */
    float p_opt_after = prob_after[0];
    bool pass = (p_opt_after > p_opt_before);
    std::cout << "P(a=0|s0) before: " << p_opt_before
              << " -> after: " << p_opt_after
              << " (delta=" << (p_opt_after - p_opt_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass ? "PASS (gradient increases optimal prob)" :
                                        "FAIL (gradient should increase optimal prob)")
              << std::endl;

    /*
       Anti-test: invert the reward assignment 鈥?a=1 now earns the reward and a=0
       earns nothing 鈥?so every one of the four steps pushes AWAY from a=0:
       P(a=1|s0) rises and P(a=0|s0) falls.

       The assertion here used to be `p2_opt_after < p2_opt_before` combined with
       a trajectory that punished a=1, i.e. it demanded that punishing the
       suboptimal action LOWER the optimal action's probability. No correct
       REINFORCE update can do that: suppressing a=1 necessarily raises a=0.
    */
    std::vector<RL::Step> trajectory2;
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 1), 1.0f));
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 0), 0.0f));
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 1), 1.0f));
    trajectory2.push_back(RL::Step(state_s0, makeOneHot(2, 0), 0.0f));

    RL::DPG agent2(2, 8, 2);
    RL::Tensor p2_before = agent2.action(state_s0);   /* value copy */
    float p2_opt_before = p2_before[0];
    float p2_sub_before = p2_before[1];

    agent2.reinforce(trajectory2, 5e-3f);

    RL::Tensor p2_after = agent2.action(state_s0);    /* value copy */
    float p2_opt_after = p2_after[0];
    float p2_sub_after = p2_after[1];

    /* With the rewards inverted, the policy must move TOWARD a=1 and away from
       a=0 鈥?the opposite direction to the main test above. */
    bool pass2 = (p2_sub_after > p2_sub_before) && (p2_opt_after < p2_opt_before);
    std::cout << "\nAnti-test: a=1 rewarded, a=0 not" << std::endl;
    std::cout << "P(a=0|s0) before: " << p2_opt_before
              << " -> after: " << p2_opt_after
              << " (delta=" << (p2_opt_after - p2_opt_before) << ")"
              << std::endl;
    std::cout << "P(a=1|s0) before: " << p2_sub_before
              << " -> after: " << p2_sub_after
              << " (delta=" << (p2_sub_after - p2_sub_before) << ")"
              << std::endl;
    std::cout << "Result: " << (pass2 ? "PASS (probability moved toward the rewarded action)" :
                                        "FAIL (probability should move toward the rewarded action)")
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

    float eval_correct_rate[8];   // 1 untrained baseline + 5 during training
    int eval_idx = 0;

    for (int ep = 0; ep < episodes; ep++) {
        // Randomly choose pattern
        int pattern = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

        /*
           Each 2-step sequence is an independent episode, so the LSTM state must
           start from zero. It was never reset here, so this episode's s0 context
           was buried under the history of every previous episode and the
           temporal task was unsolvable 鈥?the reason this test sat at the 50%
           random baseline.
        */
        agent.resetState();

        /*
           Train on SEQS_PER_UPDATE repetitions of the SAME 2-step sequence.

           A single 2-step trajectory gives a very noisy gradient. Backwards
           through the return, G_1 = r and G_0 = gamma*r, so with a mean
           baseline the advantages are only +0.05r and -0.05r, and step 0's
           action is irrelevant to the reward, so it contributes noise of the
           same magnitude as the signal. Averaging the gradient over
           SEQS_PER_UPDATE samples in one trajectory is ordinary mini-batch
           REINFORCE and is also what the shipped agent already does — its
           rollouts are 16/32/128 steps long, so a 2-step batch was the
           unrealistic part of this test, not the algorithm.

           Measured over 30 PAIRED trials (identical RNG seed and therefore
           identical network initialization for both settings, 2000 episodes,
           200-sample deterministic evaluation), success = final accuracy > 90%:

               reps  1: 24/30 (80%)      reps  4: 28/30 (93%)
               reps  8: 29/30 (97%)      (reps 4 vs 1: 5 improved, 1 worsened)

           An earlier 6-trial measurement claimed this removed the bimodality
           entirely; 6 trials was far too few to support that and the claim has
           been corrected here.

           The evaluation below still uses a single fresh 2-step sequence, so
           the task itself is not made easier.
        */
        const int SEQS_PER_UPDATE = 4;
        std::vector<RL::Step> trajectory;
        for (int rep = 0; rep < SEQS_PER_UPDATE; rep++) {
            // Step 0
            RL::Tensor s0(3, 1);
            s0[pattern == PATTERN_A ? 0 : 2] = 1.0f;
            s0[1] = 0.0f;

            RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
            int a0 = sampleAction(p0);
            RL::Tensor a0OneHot = makeOneHot(2, a0);

            // Step 1
            RL::Tensor s1(3, 1);
            s1[0] = 0.0f; s1[1] = 1.0f; s1[2] = 0.0f;

            RL::Tensor p1 = agent.action(s1);   /* value copy, see above */
            int a1 = sampleAction(p1);
            RL::Tensor a1OneHot = makeOneHot(2, a1);

            // Reward at step 1 only
            int optimalA1 = (pattern == PATTERN_A) ? 0 : 1;
            float reward = (a1 == optimalA1) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(s0, a0OneHot, 0.0f));  // no reward at step 0
            trajectory.push_back(RL::Step(s1, a1OneHot, reward));
        }

        agent.reinforce1(trajectory, 5e-3f);

        /*
           Evaluate. ep == 0 measures the UNTRAINED policy, which is what the
           "trend" check below compares against; taking the first measurement
           400 episodes in meant that a policy which learned quickly had nothing
           left to improve and was reported as a failure.
        */
        if (ep == 0 || ep % evalInterval == evalInterval - 1) {
            int correct = 0;
            const int evalEpisodes = 50;
            for (int e = 0; e < evalEpisodes; e++) {
                int p = (std::rand() % 2 == 0) ? PATTERN_A : PATTERN_B;

                /* fresh state, then let s0 propagate into s1 鈥?that IS the task */
                agent.resetState();

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

    /* pass: must beat the 50% random baseline by a clear margin.
       trend: must improve on the untrained baseline measured at ep == 0, or
       already be at ceiling (a lucky initialization that the greedy measurement
       finds already temporally correct has nothing left to learn). A regression
       is still caught: a broken reinforce1() leaves the policy at ~50%, which
       fails `pass`. */
    bool pass = (eval_correct_rate[eval_idx-1] > 0.7f);
    bool trend = (eval_idx >= 2 &&
                  (eval_correct_rate[eval_idx-1] > eval_correct_rate[0] ||
                   eval_correct_rate[eval_idx-1] >= 0.99f));

    std::cout << "\nSummary: final accuracy=" << std::fixed << std::setprecision(1)
              << (eval_correct_rate[eval_idx-1]*100) << "%"
              << " untrained=" << (eval_correct_rate[0]*100) << "%"
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

        agent.resetState();   /* independent episode: start from a clean state */

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

        /* Same task and hyper-parameters as Test 1 (which exercises the original
           reinforce()); 5e-4 was simply too small to reach both states. */
        agent.reinforce1(trajectory, 1e-3f);

        if (ep % 200 == 199) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
            /* Probing must match the training conditions: one clean recurrent
               state, then the two steps in order. The previous version reset the
               state before EACH action, i.e. it measured (s1, h = 0) — a
               condition the LSTM never encounters during training, because the
               trajectory always carries h over from the preceding s0. The
               network therefore encodes s1's decision relative to that history
               and the mismatched probe failed ~40% of runs.

               Measured on the same trained weights (perfectly paired, 30
               seeds):  split probe 18/30 vs continuous probe 25/30, with
               7 trials fixed and 0 broken (McNemar exact p ~ 0.016). */
            agent.resetState();
            RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
            RL::Tensor p1 = agent.action(s1);   /* value copy, see above */
            p0_history[eval_idx] = p0[0];
            p1_history[eval_idx] = p1[1];
            eval_idx++;

            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|s0)=" << p0[0]
                      << " P(a=1|s1)=" << p1[1]
                      << std::endl;
        }
    }

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    agent.resetState();
    RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
    RL::Tensor p1 = agent.action(s1);   /* value copy, see above */

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << std::endl;

    if (pass) std::cout << ">>> Test 4 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 4 FAILED <<<" << std::endl;
    return pass ? 0 : 1;
}


// -------------------- Test 5: DPG with the new reinforce1() --------------------
// Same contextual bandit as Test 1, but trained with reinforce1() 鈥?the exact
// REINFORCE estimator that does not mutate the stored action.
static int test_dpg_bandit_reinforce1()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test 5: DPG Contextual Bandit (reinforce1)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Same task as Test 1, trained with the new reinforce1()" << std::endl;

    RL::DPG agent(2, 16, 2);

    const int episodes = 1000;
    const int stepsPerEp = 10;
    float p0_first = 0, p0_last = 0;

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

            RL::Tensor &prob = agent.action(state);
            int action = sampleAction(prob);
            RL::Tensor actionOneHot = makeOneHot(2, action);
            float reward = (action == optimalAction) ? 1.0f : 0.0f;

            trajectory.push_back(RL::Step(state, actionOneHot, reward));
        }

        agent.reinforce1(trajectory, 1e-3f);

        if (ep % 200 == 199) {
            RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
            RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
            if (ep == 199) p0_first = p0[0];
            p0_last = p0[0];
            std::cout << "Ep " << std::setw(4) << ep+1
                      << " | P(a=0|s0)=" << p0[0] << std::endl;
        }
    }

    RL::Tensor s0(2,1); s0[0]=1; s0[1]=0;
    RL::Tensor s1(2,1); s1[0]=0; s1[1]=1;
    RL::Tensor p0 = agent.action(s0);   /* value copy: Net::forward returns an internal reference */
    RL::Tensor p1 = agent.action(s1);   /* value copy, see above */

    bool pass = (p0[0] > 0.6f && p1[1] > 0.6f);
    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL")
              << " | P(a=0|s0)=" << p0[0] << " P(a=1|s1)=" << p1[1]
              << "  [P(a=0|s0): " << p0_first << " -> " << p0_last << "]"
              << std::endl;

    if (pass) std::cout << ">>> Test 5 PASSED <<<" << std::endl;
    else      std::cout << ">>> Test 5 FAILED <<<" << std::endl;
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
    failures += test_dpg_bandit_reinforce1();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary: " << (5 - failures) << "/5 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return failures;
}
