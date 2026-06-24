#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include "rl/lstm.h"
#include "rl/loss.h"
#include "rl/activate.h"

using namespace RL;
using std::cout;
using std::endl;
using std::setw;
using std::setprecision;
using std::scientific;
using std::fixed;
using std::string;
using std::vector;
using std::min;
using std::abs;
using std::stringstream;
using std::ofstream;
using std::ifstream;

/*
 * ================================================================
 * Test Suite: LSTM (Long Short-Term Memory)
 * ================================================================
 *
 * Tests:
 *   1. test_lstm_forward_shape()         - Verify forward output shape
 *   2. test_lstm_state_propagation()     - Verify hidden state propagates through time
 *   3. test_lstm_reset_clears_state()    - Verify reset() clears cached state
 *   4. test_lstm_backward_does_not_crash()- Verify backward executes without errors
 *   5. test_lstm_sgd_updates_params()    - Verify SGD optimizer updates parameters
 *   6. test_lstm_rmsprop_updates_params()- Verify RMSProp optimizer updates parameters
 *   7. test_lstm_adam_updates_params()   - Verify Adam optimizer updates parameters
 *   8. test_lstm_learn_sine_wave()       - Verify LSTM learns sine wave sequence prediction
 *   9. test_lstm_learn_adding_problem()  - Verify LSTM learns the adding problem (memory)
 *   10. test_lstm_copy_softupdate()      - Verify copyTo and softUpdateTo work
 *   11. test_lstm_save_load()            - Verify save/load roundtrip
 *   12. test_lstm_clamp_works()          - Verify clamp limits parameter values
 * ================================================================
 */

static const float EPSILON = 1e-4f;
static const float ATOL = 1e-4f;

// -------------------- Test 1: Forward Shape --------------------
static int test_lstm_forward_shape()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 1: LSTM Forward Shape" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 4;
    const int hiddenDim = 16;
    const int outputDim = 2;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    Tensor x(inputDim, 1);
    Random::uniform(x, -1, 1);

    Tensor &out = lstm.forward(x);

    bool pass = (out.totalSize == (size_t)outputDim) &&
                (out.shape[0] == outputDim) &&
                (out.shape[1] == 1);

    cout << "Input shape: [" << inputDim << ", 1]" << endl;
    cout << "Hidden dim: " << hiddenDim << endl;
    cout << "Output shape: [" << out.shape[0] << ", " << out.shape[1] << "]" << endl;
    cout << "Output size: " << out.totalSize << " (expected " << outputDim << ")" << endl;
    cout << "Hidden size: " << lstm.h.totalSize << " (expected " << hiddenDim << ")" << endl;
    cout << "Cell size: " << lstm.c.totalSize << " (expected " << hiddenDim << ")" << endl;

    if (pass) cout << ">>> Test 1 PASSED <<<" << endl;
    else      cout << ">>> Test 1 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 2: State Propagation Through Time --------------------
static int test_lstm_state_propagation()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 2: LSTM State Propagation Through Time" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 3;
    const int hiddenDim = 8;
    const int outputDim = 2;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    Tensor x(inputDim, 1);
    Random::uniform(x, -1, 1);

    // First forward pass
    lstm.forward(x);
    Tensor h_after1 = lstm.h;
    Tensor c_after1 = lstm.c;

    // Second forward pass with same input - state should have changed
    lstm.forward(x);
    Tensor h_after2 = lstm.h;
    Tensor c_after2 = lstm.c;

    bool h_changed = false;
    for (size_t i = 0; i < h_after1.size(); i++) {
        if (abs(h_after2[i] - h_after1[i]) > EPSILON) {
            h_changed = true;
            break;
        }
    }
    bool c_changed = false;
    for (size_t i = 0; i < c_after1.size(); i++) {
        if (abs(c_after2[i] - c_after1[i]) > EPSILON) {
            c_changed = true;
            break;
        }
    }

    cout << "Hidden state changed after 2 steps: " << (h_changed ? "YES" : "NO") << endl;
    cout << "Cell state changed after 2 steps: " << (c_changed ? "YES" : "NO") << endl;

    // Now reset and verify states are zeroed
    lstm.reset();
    bool h_zeroed = true;
    for (size_t i = 0; i < lstm.h.size(); i++) {
        if (abs(lstm.h[i]) > 1e-7f) {
            h_zeroed = false;
            break;
        }
    }
    bool c_zeroed = true;
    for (size_t i = 0; i < lstm.c.size(); i++) {
        if (abs(lstm.c[i]) > 1e-7f) {
            c_zeroed = false;
            break;
        }
    }
    cout << "After reset, hidden zeroed: " << (h_zeroed ? "YES" : "NO") << endl;
    cout << "After reset, cell zeroed: " << (c_zeroed ? "YES" : "NO") << endl;

    bool pass = h_changed && c_changed && h_zeroed && c_zeroed;
    if (pass) cout << ">>> Test 2 PASSED <<<" << endl;
    else      cout << ">>> Test 2 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 3: Reset Clears State --------------------
static int test_lstm_reset_clears_state()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 3: LSTM Reset Clears Cached State" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    Tensor x(inputDim, 1);
    Random::uniform(x, -1, 1);

    // Forward should populate cacheX and states
    lstm.forward(x, false);
    lstm.forward(x, false);
    lstm.forward(x, false);

    bool has_cache_before = (lstm.cacheX.size() == 3) && (lstm.states.size() == 3);
    cout << "Has 3 cached inputs before reset: " << (has_cache_before ? "YES" : "NO") << endl;

    lstm.reset();

    bool cache_cleared = lstm.cacheX.empty() && lstm.states.empty() && lstm.cacheE.empty();
    cout << "Cache cleared after reset: " << (cache_cleared ? "YES" : "NO") << endl;

    // Also verify inference mode doesn't cache
    lstm.forward(x, true);  // inference=true
    bool inference_no_cache = lstm.cacheX.empty() && lstm.states.empty();
    cout << "Inference mode doesn't cache: " << (inference_no_cache ? "YES" : "NO") << endl;

    bool pass = has_cache_before && cache_cleared && inference_no_cache;
    if (pass) cout << ">>> Test 3 PASSED <<<" << endl;
    else      cout << ">>> Test 3 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 4: Backward Does Not Crash --------------------
static int test_lstm_backward_does_not_crash()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 4: LSTM Backward Executes Without Errors" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;
    const int seqLen = 4;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    // Forward pass for a short sequence
    vector<Tensor> inputs;
    vector<Tensor> errors;
    for (int t = 0; t < seqLen; t++) {
        Tensor x(inputDim, 1);
        Random::uniform(x, -1, 1);
        inputs.push_back(x);

        lstm.forward(x);
        Tensor e(outputDim, 1);
        e[0] = 1.0f;  // unit gradient
        errors.push_back(e);
    }

    // Run backward
    lstm.backward(inputs, errors);

    // Verify states are cleared after backward
    bool states_cleared = lstm.states.empty();
    cout << "States cleared after backward: " << (states_cleared ? "YES" : "NO") << endl;

    bool pass = states_cleared;
    if (pass) cout << ">>> Test 4 PASSED <<<" << endl;
    else      cout << ">>> Test 4 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 5: SGD Updates Parameters --------------------
static int test_lstm_sgd_updates_params()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 5: LSTM SGD Updates Parameters" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;
    const int seqLen = 3;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    // Save parameters before update
    Tensor w_before = lstm.w;
    Tensor b_before = lstm.b;
    Tensor wi_before = lstm.wi;

    // Forward + cacheError to build up gradient (correct public API pattern)
    for (int t = 0; t < seqLen; t++) {
        Tensor x(inputDim, 1);
        Random::uniform(x, -1, 1);
        lstm.forward(x);
        Tensor e(outputDim, 1);
        e[0] = 1.0f;  // constant error gradient
        lstm.cacheError(e);
    }

    // SGD internally calls backward(cacheX, cacheE), updates params, clears gradients
    lstm.SGD(0.1f);

    // Check if parameters changed
    bool w_changed = false;
    bool b_changed = false;
    bool wi_changed = false;

    for (size_t i = 0; i < lstm.w.totalSize; i++) {
        if (abs(lstm.w[i] - w_before[i]) > 1e-7f) { w_changed = true; break; }
    }
    for (size_t i = 0; i < lstm.b.totalSize; i++) {
        if (abs(lstm.b[i] - b_before[i]) > 1e-7f) { b_changed = true; break; }
    }
    for (size_t i = 0; i < lstm.wi.totalSize; i++) {
        if (abs(lstm.wi[i] - wi_before[i]) > 1e-7f) { wi_changed = true; break; }
    }

    // Verify caches are cleared after SGD
    bool caches_cleared = lstm.cacheX.empty() && lstm.cacheE.empty() && lstm.states.empty();

    cout << "w updated:     " << (w_changed ? "YES" : "NO") << endl;
    cout << "b updated:     " << (b_changed ? "YES" : "NO") << endl;
    cout << "wi updated:    " << (wi_changed ? "YES" : "NO") << endl;
    cout << "Caches cleared: " << (caches_cleared ? "YES" : "NO") << endl;

    bool pass = w_changed && b_changed && wi_changed && caches_cleared;
    if (pass) cout << ">>> Test 5 PASSED <<<" << endl;
    else      cout << ">>> Test 5 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 6: RMSProp Updates Parameters --------------------
static int test_lstm_rmsprop_updates_params()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 6: LSTM RMSProp Updates Parameters" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;
    const int seqLen = 3;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    // Save parameters before update
    Tensor w_before = lstm.w;
    Tensor b_before = lstm.b;

    // Forward + cacheError (correct public API pattern)
    for (int t = 0; t < seqLen; t++) {
        Tensor x(inputDim, 1);
        Random::uniform(x, -1, 1);
        lstm.forward(x);
        Tensor e(outputDim, 1);
        e[0] = 1.0f;
        lstm.cacheError(e);
    }

    // RMSProp internally calls backward(cacheX, cacheE), updates params
    lstm.RMSProp(0.01f, 0.9f, 0.0f, false);

    bool w_changed = false;
    bool b_changed = false;
    for (size_t i = 0; i < lstm.w.totalSize; i++) {
        if (abs(lstm.w[i] - w_before[i]) > 1e-7f) { w_changed = true; break; }
    }
    for (size_t i = 0; i < lstm.b.totalSize; i++) {
        if (abs(lstm.b[i] - b_before[i]) > 1e-7f) { b_changed = true; break; }
    }

    cout << "w updated: " << (w_changed ? "YES" : "NO") << endl;
    cout << "b updated: " << (b_changed ? "YES" : "NO") << endl;

    bool pass = w_changed && b_changed;
    if (pass) cout << ">>> Test 6 PASSED <<<" << endl;
    else      cout << ">>> Test 6 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 7: Adam Updates Parameters --------------------
static int test_lstm_adam_updates_params()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 7: LSTM Adam Updates Parameters" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;
    const int seqLen = 3;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    Tensor w_before = lstm.w;
    Tensor b_before = lstm.b;

    // Forward + cacheError (correct public API pattern)
    for (int t = 0; t < seqLen; t++) {
        Tensor x(inputDim, 1);
        Random::uniform(x, -1, 1);
        lstm.forward(x);
        Tensor e(outputDim, 1);
        e[0] = 1.0f;
        lstm.cacheError(e);
    }

    // Adam: typical hyperparameters
    lstm.Adam(0.001f, 0.9f, 0.999f, 0.9f, 0.999f, 0.0f, false);

    bool w_changed = false;
    bool b_changed = false;
    for (size_t i = 0; i < lstm.w.totalSize; i++) {
        if (abs(lstm.w[i] - w_before[i]) > 1e-7f) { w_changed = true; break; }
    }
    for (size_t i = 0; i < lstm.b.totalSize; i++) {
        if (abs(lstm.b[i] - b_before[i]) > 1e-7f) { b_changed = true; break; }
    }

    cout << "w updated: " << (w_changed ? "YES" : "NO") << endl;
    cout << "b updated: " << (b_changed ? "YES" : "NO") << endl;

    bool pass = w_changed && b_changed;
    if (pass) cout << ">>> Test 7 PASSED <<<" << endl;
    else      cout << ">>> Test 7 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 8: Learn Sine Wave Prediction --------------------
static int test_lstm_learn_sine_wave()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 8: LSTM Learn Sine Wave Sequence Prediction" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 1;
    const int hiddenDim = 16;
    const int outputDim = 1;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    // Generate sine wave data: predict next value from current value
    const int seqLen = 20;
    const int nSequences = 100;
    const int epochs = 200;

    vector<vector<Tensor>> all_inputs;
    vector<vector<Tensor>> all_targets;

    for (int s = 0; s < nSequences; s++) {
        float phase = (float)s / nSequences * 2.0f * 3.14159f;
        vector<Tensor> seq_x;
        vector<Tensor> seq_y;
        for (int t = 0; t < seqLen; t++) {
            float val = std::sin(phase + t * 0.4f);
            Tensor x(inputDim, 1);
            x[0] = val;
            Tensor y(outputDim, 1);
            y[0] = std::sin(phase + (t + 1) * 0.4f);
            seq_x.push_back(x);
            seq_y.push_back(y);
        }
        all_inputs.push_back(seq_x);
        all_targets.push_back(seq_y);
    }

    float first_loss = -1.0f;
    float last_loss = -1.0f;

    for (int ep = 0; ep < epochs; ep++) {
        // Compute loss (separate forward pass)
        float total_loss = 0;
        for (int s = 0; s < nSequences; s++) {
            lstm.reset();
            for (int t = 0; t < seqLen; t++) {
                lstm.forward(all_inputs[s][t]);
                float diff = lstm.o[0] - all_targets[s][t][0];
                total_loss += diff * diff;
            }
        }
        total_loss /= (nSequences * seqLen);

        // Training pass: forward + cacheError (then optimizer will backward+update)
        for (int s = 0; s < nSequences; s++) {
            lstm.reset();
            for (int t = 0; t < seqLen; t++) {
                lstm.forward(all_inputs[s][t]);
                Tensor e = Loss::MSE::df(lstm.o, all_targets[s][t]);
                lstm.cacheError(e);
            }
        }

        // Update with RMSProp
        lstm.RMSProp(0.01f, 0.9f, 0.0f, true);

        if (ep == 0) first_loss = total_loss;
        last_loss = total_loss;

        if (ep % 40 == 0 || ep == epochs - 1) {
            cout << "Ep " << setw(4) << ep
                 << " | loss = " << scientific << setprecision(6) << total_loss
                 << endl;
        }

        if (total_loss > 1e10f) {
            cout << "Loss exploded, aborting" << endl;
            break;
        }
    }

    bool pass = (last_loss < first_loss * 0.5f);
    cout << "First epoch loss: " << first_loss
         << ", Last epoch loss: " << last_loss
         << " (need < 50% of first)" << endl;

    // Evaluation on a test sequence
    cout << "\nEvaluation on test sequence:" << endl;
    lstm.reset();
    cout << "t\tinput\tpredict\ttarget" << endl;
    float test_phase = 0.5f;
    for (int t = 0; t < 10; t++) {
        float val = std::sin(test_phase + t * 0.4f);
        float target = std::sin(test_phase + (t + 1) * 0.4f);
        Tensor x(inputDim, 1);
        x[0] = val;
        lstm.forward(x, true);
        cout << t << "\t" << fixed << setprecision(4) << val
             << "\t" << lstm.o[0]
             << "\t" << target << endl;
    }

    if (pass) cout << ">>> Test 8 PASSED <<<" << endl;
    else      cout << ">>> Test 8 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 9: Adding Problem (Memory Test) --------------------
static int test_lstm_learn_adding_problem()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 9: LSTM Adding Problem (Memory Test)" << endl;
    cout << string(60, '=') << endl;

    /*
     * Adding Problem: Given a sequence of random numbers and two marker positions,
     * the target is the sum of the numbers at the two marker positions.
     * The LSTM must remember which values to add across many time steps.
     */
    const int inputDim = 2;    // [value, marker]
    const int hiddenDim = 20;
    const int outputDim = 1;
    const int seqLen = 10;
    const int nSequences = 200;
    const int epochs = 150;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    vector<vector<Tensor>> all_inputs;
    vector<Tensor> all_targets;

    // Generate training data
    std::srand(42);
    for (int s = 0; s < nSequences; s++) {
        vector<Tensor> seq;
        int pos1 = std::rand() % seqLen;
        int pos2 = std::rand() % seqLen;
        while (pos2 == pos1) pos2 = std::rand() % seqLen;

        Tensor target_sum(outputDim, 1);
        target_sum[0] = 0.0f;

        for (int t = 0; t < seqLen; t++) {
            Tensor xt(inputDim, 1);
            float val = (float)(std::rand() % 100) / 100.0f;
            xt[0] = val;
            if (t == pos1 || t == pos2) {
                xt[1] = 1.0f;
                target_sum[0] += val;
            } else {
                xt[1] = 0.0f;
            }
            seq.push_back(xt);
        }
        all_inputs.push_back(seq);
        all_targets.push_back(target_sum);
    }

    float first_loss = -1.0f;
    float last_loss = -1.0f;

    for (int ep = 0; ep < epochs; ep++) {
        // Compute loss
        float total_loss = 0;
        for (int s = 0; s < nSequences; s++) {
            lstm.reset();
            for (int t = 0; t < seqLen; t++) {
                lstm.forward(all_inputs[s][t]);
            }
            float diff = lstm.o[0] - all_targets[s][0];
            total_loss += diff * diff;
        }
        total_loss /= nSequences;

        if (ep == 0) first_loss = total_loss;
        last_loss = total_loss;

        // Training: forward through sequence, then cacheError at end
        for (int s = 0; s < nSequences; s++) {
            lstm.reset();
            for (int t = 0; t < seqLen; t++) {
                lstm.forward(all_inputs[s][t]);
            }
            float diff = lstm.o[0] - all_targets[s][0];
            Tensor e(outputDim, 1);
            e[0] = 2.0f * diff;  // MSE derivative
            lstm.cacheError(e);
        }

        // SGD will backward through time and update
        lstm.SGD(0.001f);

        if (ep % 30 == 0 || ep == epochs - 1) {
            cout << "Ep " << setw(4) << ep
                 << " | loss = " << scientific << setprecision(6) << total_loss
                 << endl;
        }

        if (total_loss > 1e10f) {
            cout << "Loss exploded, aborting" << endl;
            break;
        }
    }

    // Evaluate
    cout << "\nEvaluation:" << endl;
    float eval_loss = 0;
    int nEval = 20;
    for (int s = 0; s < nEval; s++) {
        lstm.reset();
        int pos1 = std::rand() % seqLen;
        int pos2 = std::rand() % seqLen;
        while (pos2 == pos1) pos2 = std::rand() % seqLen;
        float expected_sum = 0;
        for (int t = 0; t < seqLen; t++) {
            Tensor xt(inputDim, 1);
            float val = (float)(std::rand() % 100) / 100.0f;
            xt[0] = val;
            xt[1] = (t == pos1 || t == pos2) ? 1.0f : 0.0f;
            if (t == pos1 || t == pos2) expected_sum += val;
            lstm.forward(xt, true);
        }
        float err = lstm.o[0] - expected_sum;
        eval_loss += err * err;
        cout << "Sample " << s << ": predict=" << fixed << setprecision(4)
             << lstm.o[0] << " target=" << expected_sum << " err=" << err << endl;
    }
    eval_loss /= nEval;
    cout << "Eval MSE: " << eval_loss << endl;

    bool pass = (last_loss < first_loss * 0.8f);
    cout << "First epoch loss: " << first_loss
         << ", Last epoch loss: " << last_loss
         << " (need < 80% of first)" << endl;

    if (pass) cout << ">>> Test 9 PASSED <<<" << endl;
    else      cout << ">>> Test 9 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 10: Copy and SoftUpdate --------------------
static int test_lstm_copy_softupdate()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 10: LSTM Copy and SoftUpdate" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;

    LSTM lstm1(inputDim, hiddenDim, outputDim, true);
    LSTM lstm2(inputDim, hiddenDim, outputDim, true);

    // Set known values in lstm1
    for (size_t i = 0; i < lstm1.w.totalSize; i++) lstm1.w[i] = 1.5f;
    for (size_t i = 0; i < lstm1.b.totalSize; i++) lstm1.b[i] = 2.5f;
    for (size_t i = 0; i < lstm1.wi.totalSize; i++) lstm1.wi[i] = 0.5f;
    for (size_t i = 0; i < lstm1.bi.totalSize; i++) lstm1.bi[i] = -0.5f;

    // copyTo
    lstm1.copyTo(&lstm2);

    bool copy_correct = true;
    for (size_t i = 0; i < lstm2.w.totalSize; i++) {
        if (abs(lstm2.w[i] - 1.5f) > 1e-6f) { copy_correct = false; break; }
    }
    for (size_t i = 0; i < lstm2.b.totalSize; i++) {
        if (abs(lstm2.b[i] - 2.5f) > 1e-6f) { copy_correct = false; break; }
    }
    for (size_t i = 0; i < lstm2.wi.totalSize; i++) {
        if (abs(lstm2.wi[i] - 0.5f) > 1e-6f) { copy_correct = false; break; }
    }
    for (size_t i = 0; i < lstm2.bi.totalSize; i++) {
        if (abs(lstm2.bi[i] - (-0.5f)) > 1e-6f) { copy_correct = false; break; }
    }
    cout << "copyTo correct: " << (copy_correct ? "YES" : "NO") << endl;

    // Test softUpdateTo: dst = (1-rho)*dst + rho*src
    LSTM lstm3(inputDim, hiddenDim, outputDim, true);
    for (size_t i = 0; i < lstm3.w.totalSize; i++) lstm3.w[i] = 0.0f;
    for (size_t i = 0; i < lstm1.w.totalSize; i++) lstm1.w[i] = 2.0f;

    lstm1.softUpdateTo(&lstm3, 0.5f);

    bool soft_correct = true;
    for (size_t i = 0; i < lstm3.w.totalSize; i++) {
        if (abs(lstm3.w[i] - 1.0f) > 0.01f) { soft_correct = false; break; }
    }
    cout << "softUpdateTo correct: " << (soft_correct ? "YES" : "NO") << endl;

    bool pass = copy_correct && soft_correct;
    if (pass) cout << ">>> Test 10 PASSED <<<" << endl;
    else      cout << ">>> Test 10 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 11: Save/Load Roundtrip --------------------
static int test_lstm_save_load()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 11: LSTM Save/Load Roundtrip" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 6;
    const int outputDim = 1;

    LSTM lstm1(inputDim, hiddenDim, outputDim, true);
    LSTM lstm2(inputDim, hiddenDim, outputDim, true);

    // Write to string stream
    stringstream ss;
    lstm1.write((ofstream&)ss);

    // Read back
    lstm2.read((ifstream&)ss);

    // Compare all parameters
    auto compareTensor = [](const Tensor &a, const Tensor &b, const string &name) -> bool {
        bool match = true;
        for (size_t i = 0; i < a.totalSize; i++) {
            if (abs(a[i] - b[i]) > ATOL) {
                if (match) {
                    cout << name << " mismatch at [" << i << "]: "
                         << a[i] << " vs " << b[i] << endl;
                }
                match = false;
            }
        }
        return match;
    };

    bool wi_match = compareTensor(lstm1.wi, lstm2.wi, "wi");
    bool wf_match = compareTensor(lstm1.wf, lstm2.wf, "wf");
    bool wg_match = compareTensor(lstm1.wg, lstm2.wg, "wg");
    bool wo_match = compareTensor(lstm1.wo, lstm2.wo, "wo");
    bool ui_match = compareTensor(lstm1.ui, lstm2.ui, "ui");
    bool uf_match = compareTensor(lstm1.uf, lstm2.uf, "uf");
    bool ug_match = compareTensor(lstm1.ug, lstm2.ug, "ug");
    bool uo_match = compareTensor(lstm1.uo, lstm2.uo, "uo");
    bool bi_match = compareTensor(lstm1.bi, lstm2.bi, "bi");
    bool bf_match = compareTensor(lstm1.bf, lstm2.bf, "bf");
    bool bg_match = compareTensor(lstm1.bg, lstm2.bg, "bg");
    bool bo_match = compareTensor(lstm1.bo, lstm2.bo, "bo");
    bool w_match  = compareTensor(lstm1.w, lstm2.w, "w");
    bool b_match  = compareTensor(lstm1.b, lstm2.b, "b");

    bool pass = wi_match && wf_match && wg_match && wo_match &&
                ui_match && uf_match && ug_match && uo_match &&
                bi_match && bf_match && bg_match && bo_match &&
                w_match && b_match;

    cout << "\nParameter comparison:" << endl;
    cout << "wi: " << (wi_match ? "OK" : "FAIL") << endl;
    cout << "wf: " << (wf_match ? "OK" : "FAIL") << endl;
    cout << "wg: " << (wg_match ? "OK" : "FAIL") << endl;
    cout << "wo: " << (wo_match ? "OK" : "FAIL") << endl;
    cout << "ui: " << (ui_match ? "OK" : "FAIL") << endl;
    cout << "uf: " << (uf_match ? "OK" : "FAIL") << endl;
    cout << "ug: " << (ug_match ? "OK" : "FAIL") << endl;
    cout << "uo: " << (uo_match ? "OK" : "FAIL") << endl;
    cout << "bi: " << (bi_match ? "OK" : "FAIL") << endl;
    cout << "bf: " << (bf_match ? "OK" : "FAIL") << endl;
    cout << "bg: " << (bg_match ? "OK" : "FAIL") << endl;
    cout << "bo: " << (bo_match ? "OK" : "FAIL") << endl;
    cout << "w:  " << (w_match ? "OK" : "FAIL") << endl;
    cout << "b:  " << (b_match ? "OK" : "FAIL") << endl;

    if (pass) cout << "\n>>> Test 11 PASSED <<<" << endl;
    else      cout << "\n>>> Test 11 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Test 12: Clamp Works --------------------
static int test_lstm_clamp_works()
{
    cout << "\n" << string(60, '=') << endl;
    cout << "Test 12: LSTM Clamp Limits Parameter Values" << endl;
    cout << string(60, '=') << endl;

    const int inputDim = 2;
    const int hiddenDim = 8;
    const int outputDim = 1;

    LSTM lstm(inputDim, hiddenDim, outputDim, true);

    // Set all params to large values
    for (size_t i = 0; i < lstm.w.totalSize; i++) lstm.w[i] = 100.0f;
    for (size_t i = 0; i < lstm.b.totalSize; i++) lstm.b[i] = -100.0f;
    for (size_t i = 0; i < lstm.wi.totalSize; i++) lstm.wi[i] = 200.0f;

    lstm.clamp(-1.0f, 1.0f);

    bool w_clamped = true;
    bool b_clamped = true;
    bool wi_clamped = true;

    for (size_t i = 0; i < lstm.w.totalSize; i++) {
        if (lstm.w[i] < -1.0f - 1e-6f || lstm.w[i] > 1.0f + 1e-6f) { w_clamped = false; break; }
    }
    for (size_t i = 0; i < lstm.b.totalSize; i++) {
        if (lstm.b[i] < -1.0f - 1e-6f || lstm.b[i] > 1.0f + 1e-6f) { b_clamped = false; break; }
    }
    for (size_t i = 0; i < lstm.wi.totalSize; i++) {
        if (lstm.wi[i] < -1.0f - 1e-6f || lstm.wi[i] > 1.0f + 1e-6f) { wi_clamped = false; break; }
    }

    cout << "w  clamped:  " << (w_clamped ? "YES" : "NO")
         << " (range: [" << lstm.w.min() << ", " << lstm.w.max() << "])" << endl;
    cout << "b  clamped:  " << (b_clamped ? "YES" : "NO")
         << " (range: [" << lstm.b.min() << ", " << lstm.b.max() << "])" << endl;
    cout << "wi clamped:  " << (wi_clamped ? "YES" : "NO")
         << " (range: [" << lstm.wi.min() << ", " << lstm.wi.max() << "])" << endl;

    bool pass = w_clamped && b_clamped && wi_clamped;
    if (pass) cout << ">>> Test 12 PASSED <<<" << endl;
    else      cout << ">>> Test 12 FAILED <<<" << endl;
    return pass ? 0 : 1;
}

// -------------------- Main --------------------
int main()
{
    cout << "=== LSTM (Long Short-Term Memory) Test Suite ===" << endl;
    cout << "Date: " << __DATE__ << " " << __TIME__ << endl;
    cout << "Tests for LSTM cell: forward, backward, optimizers, learning" << endl;

    int failures = 0;

    failures += test_lstm_forward_shape();
    failures += test_lstm_state_propagation();
    failures += test_lstm_reset_clears_state();
    failures += test_lstm_backward_does_not_crash();
    failures += test_lstm_sgd_updates_params();
    failures += test_lstm_rmsprop_updates_params();
    failures += test_lstm_adam_updates_params();
    failures += test_lstm_learn_sine_wave();
    failures += test_lstm_learn_adding_problem();
    failures += test_lstm_copy_softupdate();
    failures += test_lstm_save_load();
    failures += test_lstm_clamp_works();

    cout << "\n" << string(60, '=') << endl;
    cout << "Summary: " << (12 - failures) << "/12 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " FAILED)" : "")
              << endl;
    cout << string(60, '=') << endl;

    return failures;
}
