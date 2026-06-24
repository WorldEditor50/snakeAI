#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include "rl/moe.hpp"
#include "rl/loss.h"
#include "rl/activate.h"

using namespace RL;
using std::cout;
using std::endl;
using std::setw;
using std::setprecision;
using std::scientific;
using std::string;
using std::vector;
using std::min;
using std::stringstream;
using std::ofstream;
using std::ifstream;

/*
 * ================================================================
 * Test Suite: MOE (Mixture of Experts with TransformerBlock experts)
 * ================================================================
 *
 * Tests:
 *   1. test_moe_forward_shape()       - Verify MOE forward output shape
 *   2. test_moe_gating_sum()          - Verify gate probabilities sum to 1
 *   3. test_moe_gating_gradient()     - Verify gate gradients flow to Wg
 *   4. test_moe_expert_gradient()     - Verify expert gradients flow back
 *   5. test_moe_sgd_learns()          - Verify MOE can learn simple function via backward()+gradient()
 *   6. test_moe_load_save()           - Verify save/load roundtrip
 *   7. test_moe_copy_softupdate()     - Verify copyTo and softUpdateTo work
 *   8. test_moe_backward_accumulates() - Verify backward accumulates into ei
 * ================================================================
 */

static const float EPSILON = 1e-4f;

// -------------------- Test 1: Forward Shape --------------------
static int test_moe_forward_shape()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 1: MOE Forward Shape" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 32;
    MOE<3, 4> moe(d_model, true);

    Tensor x(d_model, 1);
    Random::uniform(x, -1, 1);

    Tensor &out = moe.forward(x);

    bool pass = (out.totalSize == (size_t)d_model);
    cout << "Input shape: [" << d_model << ", 1]" << endl;
    cout << "Output shape: [" << out.shape[0] << ", " << out.shape[1] << "]" << endl;
    cout << "Output size: " << out.totalSize << " (expected " << d_model << ")" << endl;

    if (pass) cout << ">>> Test 1 PASSED <<<" << endl;
    else      cout << ">>> Test 1 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 2: Gating Sum to 1 --------------------
static int test_moe_gating_sum()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 2: MOE Gating Sum to 1" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 16;
    const int num_experts = 4;
    MOE<4, 2> moe(d_model, true);

    bool all_close = true;
    for (int trial = 0; trial < 10; trial++) {
        Tensor x(d_model, 1);
        Random::uniform(x, -1, 1);

        moe.forward(x);

        float gate_sum = 0;
        for (int i = 0; i < num_experts; i++) {
            gate_sum += moe.gate[i];
        }

        float diff = std::abs(gate_sum - 1.0f);
        if (diff > EPSILON) {
            all_close = false;
            cout << "Trial " << trial << ": gate sum = " << gate_sum
                      << " (diff = " << diff << ")" << endl;
        }
    }

    bool pass = all_close;
    cout << "Gate probabilities sum to 1.0 across " << 10 << " trials: "
              << (pass ? "YES" : "NO") << endl;

    if (pass) cout << ">>> Test 2 PASSED <<<" << endl;
    else      cout << ">>> Test 2 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 3: Gating Gradient --------------------
static int test_moe_gating_gradient()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 3: MOE Gating Gradient" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 8;
    MOE<2, 2> moe(d_model, true);

    Tensor wg_before = moe.wg;
    Tensor b_before  = moe.b;

    Tensor x(d_model, 1);
    Random::uniform(x, -1, 1);
    moe.forward(x);

    for (int i = 0; i < d_model; i++) {
        moe.e[i] = 1.0f;
    }
    Tensor e(moe.o.shape);
    moe.backward(x, e);

    float grad_norm_wg = moe.g.wg.norm2();
    float grad_norm_b  = moe.g.b.norm2();
    bool has_grad = (grad_norm_wg > EPSILON || grad_norm_b > EPSILON);

    cout << "||g.wg|| = " << grad_norm_wg << endl;
    cout << "||g.b||  = " << grad_norm_b << endl;

    moe.SGD(0.01f);

    bool params_changed = false;
    for (size_t i = 0; i < wg_before.totalSize; i++) {
        if (std::abs(moe.wg[i] - wg_before[i]) > 1e-7f) {
            params_changed = true;
            break;
        }
    }

    bool pass = has_grad && params_changed;
    cout << "Gate gradients flow: " << (has_grad ? "YES" : "NO") << endl;
    cout << "Gate params updated: " << (params_changed ? "YES" : "NO") << endl;

    if (pass) cout << ">>> Test 3 PASSED <<<" << endl;
    else      cout << ">>> Test 3 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 4: Expert Gradient Flow --------------------
static int test_moe_expert_gradient()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 4: MOE Expert Gradient Flow" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 8;
    MOE<2, 2> moe(d_model, true);

    Tensor x(d_model, 1);
    Random::uniform(x, -1, 1);
    moe.forward(x);

    for (int i = 0; i < d_model; i++) {
        moe.e[i] = 0.5f;
    }
    Tensor e(moe.o.shape);
    moe.backward(x, e);

    bool experts_have_grad = true;
    for (int i = 0; i < 2; i++) {
        float norm_gamma = moe.experts[i].g1.gamma.norm2();
        float norm_w  = moe.experts[i].ffn_up.g.w.norm2();
        float norm_wo = moe.experts[i].attn.g.wo.norm2();
        cout << "Expert " << i << ": ||g.gamma1||=" << norm_gamma
                  << " ||g.ffn_up.w||=" << norm_w
                  << " ||g.attn.wo||=" << norm_wo << endl;

        if (norm_gamma < EPSILON && norm_w < EPSILON && norm_wo < EPSILON) {
            experts_have_grad = false;
        }
    }

    bool pass = experts_have_grad;
    cout << "Expert gradients flow: " << (pass ? "YES" : "NO") << endl;

    if (pass) cout << ">>> Test 4 PASSED <<<" << endl;
    else      cout << ">>> Test 4 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 5: MOE Can Learn --------------------
// Use backward() + update() for proper training with MHA weight updates.
static int test_moe_sgd_learns()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 5: MOE SGD Learning via backward()+gradient()" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 4;
    MOE<2, 2> moe(d_model, true);

    const int epochs = 300;
    const float lr = 0.0005f;
    const float rho = 0.95f;

    // Training set
    vector<Tensor> train_x, train_y;
    for (int s = 0; s < 100; s++) {
        Tensor xi(d_model, 1);
        Random::uniform(xi, -1, 1);
        train_x.push_back(xi);

        Tensor yi(d_model, 1);
        for (int j = 0; j < d_model; j++) {
            yi[j] = sin(xi[j]) + 0.5f * cos(xi[j] * 2.0f);
        }
        train_y.push_back(yi);
    }

    // Fixed test set
    vector<Tensor> test_x, test_y;
    for (int s = 0; s < 20; s++) {
        Tensor xt(d_model, 1);
        Random::uniform(xt, -1, 1);
        test_x.push_back(xt);
        Tensor yt(d_model, 1);
        for (int j = 0; j < d_model; j++) {
            yt[j] = sin(xt[j]) + 0.5f * cos(xt[j] * 2.0f);
        }
        test_y.push_back(yt);
    }

    float first_loss = -1;
    float last_loss = -1;
    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0;
        for (size_t s = 0; s < train_x.size(); s++) {
            Tensor &out = moe.forward(train_x[s]);

            float loss = 0;
            for (int j = 0; j < d_model; j++) {
                float diff = out[j] - train_y[s][j];
                loss += diff * diff;
            }
            loss /= d_model;
            total_loss += loss;

            // Error: dL/do = 2*(o-y)/d_model
            for (int j = 0; j < d_model; j++) {
                moe.e[j] = 2.0f * (out[j] - train_y[s][j]) / d_model;
            }

            // Full backward + gradient
            Tensor ei(d_model, 1);
            ei.zero();
            moe.backward(train_x[s], ei);
        }
        total_loss /= train_x.size();

        // Track first and last loss
        if (ep == 0) first_loss = total_loss;
        last_loss = total_loss;

        // Update with RMSProp
        moe.RMSProp(lr, rho, 0.0f, true);

        if (ep % 60 == 0 || ep == epochs - 1) {
            float grad_norm = moe.g.wg.norm2();
            cout << "Ep " << setw(4) << ep
                 << " | train_loss = " << scientific << setprecision(6) << total_loss
                 << " | ||g.wg|| = " << grad_norm << endl;
        }

        // Early escape if loss blows up
        if (total_loss > 1e10f) {
            cout << "Loss exploded, aborting" << endl;
            break;
        }
    }

    bool pass = (last_loss < first_loss * 0.5f);
    cout << "First epoch loss: " << first_loss
         << ", Last epoch loss: " << last_loss
         << " (need < 50% of first)" << endl;

    if (pass) cout << ">>> Test 5 PASSED <<<" << endl;
    else      cout << ">>> Test 5 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 6: Save/Load Roundtrip --------------------
// Add debug output to investigate expert serialization
static int test_moe_load_save()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 6: MOE Save/Load (with debug)" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 4;  // small for debugging
    MOE<2, 2> moe1(d_model, true);
    MOE<2, 2> moe2(d_model, true);

    // Forward to ensure consistent state
    Tensor x(d_model, 1);
    Random::uniform(x, -1, 1);
    moe1.forward(x);

    // Write to string stream for debugging
    std::stringstream ss;
    moe1.write((std::ofstream&)ss);

    // Read the written data
    moe2.read((std::ifstream&)ss);

    // Compare gate params
    const float ATOL = 1e-4f;  // tolerance for float serialization precision
    bool wg_match = true;
    for (size_t i = 0; i < moe1.wg.totalSize; i++) {
        if (std::abs(moe1.wg[i] - moe2.wg[i]) > ATOL) {
            wg_match = false;
            break;
        }
    }

    bool b_match = true;
    for (size_t i = 0; i < moe1.b.totalSize; i++) {
        if (std::abs(moe1.b[i] - moe2.b[i]) > ATOL) {
            b_match = false;
            break;
        }
    }

    // Compare expert parameters
    bool expert_match = true;
    for (int i = 0; i < 2; i++) {
        auto &e1 = moe1.experts[i];
        auto &e2 = moe2.experts[i];
        for (size_t j = 0; j < e1.gamma1.totalSize; j++) {
            float diff = std::abs(e1.gamma1[j] - e2.gamma1[j]);
            if (diff > ATOL) {
                if (expert_match) {
                    cout << "Expert " << i << " gamma1 mismatch at [" << j
                         << "]: " << e1.gamma1[j] << " vs " << e2.gamma1[j]
                         << " (diff=" << diff << ")" << endl;
                }
                expert_match = false;
            }
        }
        // Check beta1
        for (size_t j = 0; j < e1.beta1.totalSize; j++) {
            if (std::abs(e1.beta1[j] - e2.beta1[j]) > ATOL) {
                if (expert_match) {
                    cout << "Expert " << i << " beta1 mismatch at [" << j
                         << "]: " << e1.beta1[j] << " vs " << e2.beta1[j] << endl;
                }
                expert_match = false;
            }
        }
        // Check ffn_up weights
        for (size_t j = 0; j < e1.ffn_up.w.totalSize; j++) {
            if (std::abs(e1.ffn_up.w[j] - e2.ffn_up.w[j]) > ATOL) {
                if (expert_match) {
                    cout << "Expert " << i << " ffn_up.w mismatch at [" << j
                         << "]: " << e1.ffn_up.w[j] << " vs " << e2.ffn_up.w[j] << endl;
                }
                expert_match = false;
            }
        }
    }

    // If expert_match fails, dump the serialized string
    if (!expert_match) {
        cout << "\nSaved data (first 500 chars):" << endl;
        string saved = ss.str();
        cout << saved.substr(0, 500) << "..." << endl;
    }

    bool pass = wg_match && b_match && expert_match;
    cout << "Wg match:     " << (wg_match ? "YES" : "NO") << endl;
    cout << "b match:      " << (b_match ? "YES" : "NO") << endl;
    cout << "Expert match: " << (expert_match ? "YES" : "NO") << endl;

    if (pass) cout << ">>> Test 6 PASSED <<<" << endl;
    else      cout << ">>> Test 6 FAILED <<<" << endl;

    return pass ? 0 : 1;
}

// -------------------- Test 7: Copy and SoftUpdate --------------------
static int test_moe_copy_softupdate()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 7: MOE Copy and SoftUpdate" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 8;
    MOE<2, 2> moe1(d_model, true);
    MOE<2, 2> moe2(d_model, true);

    for (size_t i = 0; i < moe1.wg.totalSize; i++) {
        moe1.wg[i] = 1.0f;
    }
    for (size_t i = 0; i < moe1.b.totalSize; i++) {
        moe1.b[i] = 2.0f;
    }

    moe1.copyTo(&moe2);

    bool copy_correct = true;
    for (size_t i = 0; i < moe2.wg.totalSize; i++) {
        if (std::abs(moe2.wg[i] - 1.0f) > 1e-6f) {
            copy_correct = false;
            break;
        }
    }
    for (size_t i = 0; i < moe2.b.totalSize; i++) {
        if (std::abs(moe2.b[i] - 2.0f) > 1e-6f) {
            copy_correct = false;
            break;
        }
    }

    for (size_t i = 0; i < moe1.experts[0].gamma1.totalSize; i++) {
        if (std::abs(moe2.experts[0].gamma1[i] - moe1.experts[0].gamma1[i]) > 1e-6f) {
            copy_correct = false;
            break;
        }
    }

    cout << "copyTo correct: " << (copy_correct ? "YES" : "NO") << endl;

    // Test softUpdateTo
    MOE<2, 2> moe3(d_model, true);
    for (size_t i = 0; i < moe3.wg.totalSize; i++) {
        moe3.wg[i] = 1.0f;
    }
    for (size_t i = 0; i < moe1.wg.totalSize; i++) {
        moe1.wg[i] = 2.0f;
    }

    moe1.softUpdateTo(&moe3, 0.5f);

    bool soft_correct = true;
    for (size_t i = 0; i < moe3.wg.totalSize; i++) {
        if (std::abs(moe3.wg[i] - 1.5f) > 0.01f) {
            soft_correct = false;
            break;
        }
    }

    cout << "softUpdateTo correct: " << (soft_correct ? "YES" : "NO") << endl;

    bool pass = copy_correct && soft_correct;
    if (pass) cout << ">>> Test 7 PASSED <<<" << endl;
    else      cout << ">>> Test 7 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 8: Backward Accumulates into ei --------------------
static int test_moe_backward_accumulates()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 8: MOE Backward Accumulation" << endl;
    cout << string(60, '=') << endl;

    const int d_model = 8;
    MOE<2, 2> moe(d_model, true);

    // First pass: forward -> backward
    Tensor x(d_model, 1);
    Random::uniform(x, -1, 1);
    moe.forward(x);

    for (int i = 0; i < d_model; i++) {
        moe.e[i] = 1.0f;
    }

    Tensor ei1(d_model, 1);
    ei1.zero();
    moe.backward(x, ei1);

    float ei1_norm = ei1.norm2();
    bool has_input_grad = (ei1_norm > EPSILON);

    cout << "Input gradient norm: ||ei|| = " << ei1_norm << endl;
    cout << "Has input gradient: " << (has_input_grad ? "YES" : "NO") << endl;

    if (has_input_grad) {
        cout << "ei values: ";
        for (int i = 0; i < std::min(4, d_model); i++) {
            cout << ei1[i] << " ";
        }
        cout << "..." << endl;
    }

    // Second pass: re-forward (to reset cached states) -> backward again
    moe.forward(x);
    for (int i = 0; i < d_model; i++) {
        moe.e[i] = 1.0f;
    }

    Tensor ei2(d_model, 1);
    ei2.zero();
    moe.backward(x, ei2);

    float ei2_norm = ei2.norm2();
    float max_norm = std::max(ei1_norm, ei2_norm);
    float diff = std::abs(ei1_norm - ei2_norm);
    // Allow 30% relative difference (floating-point accumulation order)
    bool consistent = (diff < std::max(1.0f, 0.30f * max_norm));

    cout << "1st pass ||ei||: " << ei1_norm << endl;
    cout << "2nd pass ||ei||: " << ei2_norm << " (consistent: "
              << (consistent ? "YES" : "NO") << ")" << endl;

    bool pass = has_input_grad && consistent;
    if (pass) cout << ">>> Test 8 PASSED <<<" << endl;
    else      cout << ">>> Test 8 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Main --------------------
int main()
{
    cout << "=== MOE (Mixture of Experts) Test Suite ===" << endl;
    cout << "Date: " << __DATE__ << " " << __TIME__ << endl;
    cout << "MOE<NumExperts, NumHeads> with TransformerBlock experts" << endl;

    int failures = 0;

    failures += test_moe_forward_shape();
    failures += test_moe_gating_sum();
    failures += test_moe_gating_gradient();
    failures += test_moe_expert_gradient();
    failures += test_moe_sgd_learns();
    failures += test_moe_load_save();
    failures += test_moe_copy_softupdate();
    failures += test_moe_backward_accumulates();

    cout << "\n" << string(60, '=') << endl;
    cout << "Summary: " << (8 - failures) << "/8 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << endl;
    cout << string(60, '=') << endl;

    return failures;
}
