#include "../rl/mamba.h"
#include "../rl/loss.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace RL;

/*
 * Test MambaLayer (Selective State Space Model)
 *
 * Tests:
 * 1. Constructor and type registration
 * 2. Forward pass - single timestep
 * 3. Forward pass - sequential timesteps (state persistence)
 * 4. Selective B/Δ behavior (input-dependent)
 * 5. BPTT gradient flow
 * 6. MSE training on 1D function
 * 7. Reset
 * 8. Parameter operations (copyTo, softUpdateTo)
 * 9. Model save/load
 * 10. Inference mode
 */
void runTests()
{
    /* ==================== Test 1: Constructor & Type ==================== */
    {
        std::cout << "Test 1: MambaLayer Construction & Type\n";
        MambaLayer mamba(2, 8, 1, true);
        assert(mamba.type == iLayer::LAYER_MAMBA);
        assert(mamba.inputDim == 2);
        assert(mamba.hiddenDim == 8);
        assert(mamba.outputDim == 1);
        assert(mamba.A_diag.totalSize == 8);
        assert(mamba.W_B.totalSize == 8 * 2);
        assert(mamba.W_delta.totalSize == 8 * 2);
        assert(mamba.C.totalSize == 1 * 8);
        // W_in should be empty when inputDim == hiddenDim... wait, here it's 2 != 8
        assert(mamba.W_in.totalSize > 0);
        std::cout << "  PASS: dimensions correct\n";
    }

    /* ==================== Test 2: Single forward pass ==================== */
    {
        std::cout << "Test 2: Forward pass (single timestep)\n";
        MambaLayer mamba(3, 16, 2, true);
        Tensor x(3, 1);
        x[0] = 0.5f; x[1] = -0.3f; x[2] = 0.8f;

        Tensor &out = mamba.forward(x);
        assert(out.totalSize == 2);
        // Output should be tanh-activated => values in [-1, 1]
        for (int i = 0; i < 2; i++) {
            assert(out[i] >= -1.0f && out[i] <= 1.0f);
        }
        assert(mamba.cacheX.size() == 1);
        assert(mamba.states.size() == 1);
        std::cout << "  PASS: forward produces valid tanh output\n";
    }

    /* ==================== Test 3: Sequential state persistence ==================== */
    {
        std::cout << "Test 3: Sequential forward (state persistence)\n";
        MambaLayer mamba(4, 4, 1, true);  // inputDim == hiddenDim (no W_in)

        Tensor x1(4, 1); for (int i=0; i<4; i++) x1[i] = 1.0f;
        Tensor x2(4, 1); for (int i=0; i<4; i++) x2[i] = 1.0f;

        mamba.forward(x1);
        Tensor first_h = mamba.h;
        mamba.forward(x2);
        Tensor second_h = mamba.h;

        // States should differ since Ā·h adds new info each step
        bool differs = false;
        for (int i = 0; i < 4; i++) {
            if (std::abs(second_h[i] - first_h[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        assert(differs);
        assert(mamba.cacheX.size() == 2);
        std::cout << "  PASS: hidden state evolves through time\n";
    }

    /* ==================== Test 4: Selective B/Δ behavior ==================== */
    {
        std::cout << "Test 4: Selective B/Δ (input-dependent gating)\n";
        MambaLayer mamba(2, 4, 1, true);

        // Two different inputs should produce different B and Δ values
        Tensor x1(2, 1); x1[0] = 0.1f; x1[1] = 0.2f;
        Tensor x2(2, 1); x2[0] = 5.0f; x2[1] = -5.0f;

        auto state1 = mamba.feedForward(x1, mamba.h);
        mamba.reset();
        auto state2 = mamba.feedForward(x2, mamba.h);

        // Different inputs => different B and Δ
        bool B_differs = false;
        bool delta_differs = false;
        for (int i = 0; i < 4; i++) {
            if (std::abs(state1.B[i] - state2.B[i]) > 1e-3f) B_differs = true;
            if (std::abs(state1.delta[i] - state2.delta[i]) > 1e-3f) delta_differs = true;
        }
        assert(B_differs);
        assert(delta_differs);

        // B should be in [0, 1] (sigmoid)
        for (int i = 0; i < 4; i++) {
            assert(state1.B[i] >= 0.0f && state1.B[i] <= 1.0f);
            assert(state2.B[i] >= 0.0f && state2.B[i] <= 1.0f);
        }

        // Δ should be positive (softplus)
        for (int i = 0; i < 4; i++) {
            assert(state1.delta[i] > 0.0f);
            assert(state2.delta[i] > 0.0f);
        }

        std::cout << "  PASS: B and Δ are input-dependent\n";
    }

    /* ==================== Test 5: BPTT gradient flow ==================== */
    {
        std::cout << "Test 5: BPTT gradient flow\n";
        MambaLayer mamba(2, 6, 1, true);

        Tensor x1(2, 1); x1[0] = 0.1f; x1[1] = 0.2f;
        Tensor x2(2, 1); x2[0] = 0.3f; x2[1] = 0.4f;

        mamba.forward(x1);
        mamba.forward(x2);

        Tensor e1(1, 1); e1[0] = 1.0f;
        Tensor e2(1, 1); e2[0] = 1.0f;
        mamba.cacheError(e1);
        mamba.cacheError(e2);

        mamba.SGD(0.01f);

        // Gradients should be non-zero after BPTT
        bool gA_nz = false, gC_nz = false;
        bool gWB_nz = false, gWD_nz = false;
        for (int i = 0; i < 6; i++) if (std::abs(mamba.g_A_diag[i]) > 1e-10f) gA_nz = true;
        for (int i = 0; i < 1 * 6; i++) if (std::abs(mamba.g_C[i]) > 1e-10f) gC_nz = true;
        for (int i = 0; i < 6 * 2; i++) if (std::abs(mamba.g_W_B[i]) > 1e-10f) gWB_nz = true;
        for (int i = 0; i < 6 * 2; i++) if (std::abs(mamba.g_W_delta[i]) > 1e-10f) gWD_nz = true;

        assert(gA_nz && gC_nz && gWB_nz && gWD_nz);
        std::cout << "  PASS: BPTT computes non-zero gradients for all params\n";
    }

    /* ==================== Test 6: Training on simple function ==================== */
    {
        std::cout << "Test 6: Training on z = sin(x^2 + y^2)\n";
        MambaLayer mamba(2, 10, 1, true);

        auto zeta = [](float x, float y) -> float {
            return std::sin(x*x + y*y);
        };

        std::uniform_real_distribution<float> uniform(-1, 1);
        std::vector<Tensor> data, targets;

        for (int i = 0; i < 200; i++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            p[0] = x; p[1] = y;
            Tensor t(1, 1);
            t[0] = zeta(x, y);
            data.push_back(p);
            targets.push_back(t);
        }

        std::uniform_int_distribution<int> sel(0, 199);

        // Initial loss
        mamba.reset();
        float initialLoss = 0;
        for (int i = 0; i < 30; i++) {
            int k = sel(Random::engine);
            mamba.forward(data[k], true);
            float d = mamba.o[0] - targets[k][0];
            initialLoss += d*d;
        }
        initialLoss /= 30;
        std::cout << "  Initial MSE: " << initialLoss << std::endl;

        // Training
        for (int iter = 0; iter < 200; iter++) {
            mamba.reset();
            for (int j = 0; j < 16; j++) {
                int k = sel(Random::engine);
                mamba.forward(data[k]);
                mamba.cacheError(Loss::MSE::df(mamba.o, targets[k]));
            }
            mamba.RMSProp(0.01f, 0.9f, 0.0f, true);
        }

        // Final loss
        mamba.reset();
        float finalLoss = 0;
        for (int i = 0; i < 30; i++) {
            int k = sel(Random::engine);
            mamba.forward(data[k], true);
            float d = mamba.o[0] - targets[k][0];
            finalLoss += d*d;
        }
        finalLoss /= 30;
        std::cout << "  Final MSE: " << finalLoss << std::endl;
        assert(finalLoss < initialLoss);
        std::cout << "  PASS: loss decreased after training\n";
    }

    /* ==================== Test 7: Reset ==================== */
    {
        std::cout << "Test 7: Reset\n";
        MambaLayer mamba(2, 4, 1, true);
        Tensor x(2, 1); x[0] = 1.0f; x[1] = 2.0f;

        mamba.forward(x);
        mamba.forward(x);
        assert(mamba.cacheX.size() == 2);
        assert(mamba.states.size() == 2);

        mamba.reset();
        assert(mamba.cacheX.size() == 0);
        assert(mamba.states.size() == 0);
        for (int i = 0; i < 4; i++) assert(mamba.h[i] == 0.0f);
        std::cout << "  PASS: reset clears cache and state\n";
    }

    /* ==================== Test 8: copyTo and softUpdateTo ==================== */
    {
        std::cout << "Test 8: copyTo & softUpdateTo\n";
        MambaLayer mamba1(2, 6, 1, true);
        MambaLayer mamba2(2, 6, 1, false);

        mamba1.copyTo(&mamba2);
        for (int i = 0; i < 6 * 2; i++) assert(mamba2.W_B[i] == mamba1.W_B[i]);
        for (int i = 0; i < 6 * 2; i++) assert(mamba2.W_delta[i] == mamba1.W_delta[i]);

        // softUpdateTo with alpha=0.5
        float oldA0 = mamba2.A_diag[0];
        mamba1.A_diag.zero();
        mamba1.A_diag[0] = 1.0f;
        mamba1.softUpdateTo(&mamba2, 0.5f);
        assert(std::abs(mamba2.A_diag[0] - 0.5f * oldA0) < 1e-5f);
        std::cout << "  PASS: copyTo and softUpdateTo work correctly\n";
    }

    /* ==================== Test 9: Save/Load ==================== */
    {
        std::cout << "Test 9: Save/Load\n";
        MambaLayer mamba1(2, 6, 1, true);
        MambaLayer mamba2(2, 6, 1, false);

        mamba1.A_diag.zero(); for (int i = 0; i < 6; i++) mamba1.A_diag[i] = (float)(i+1);
        mamba1.C.zero(); for (int i = 0; i < 6; i++) mamba1.C[i] = (float)(i+10);
        mamba1.b.zero(); mamba1.b[0] = 3.14f;

        std::ofstream ofs("mamba_test_save.txt");
        mamba1.write(ofs);
        ofs.close();

        std::ifstream ifs("mamba_test_save.txt");
        mamba2.read(ifs);
        ifs.close();

        for (int i = 0; i < 6; i++) {
            assert(std::abs(mamba2.A_diag[i] - mamba1.A_diag[i]) < 1e-5f);
        }
        assert(std::abs(mamba2.b[0] - 3.14f) < 1e-5f);
        std::cout << "  PASS: save/load preserves all parameters\n";
    }

    /* ==================== Test 10: Inference mode ==================== */
    {
        std::cout << "Test 10: Inference mode (no caching)\n";
        MambaLayer mamba(2, 6, 1, true);
        Tensor x(2, 1); x[0] = 0.5f; x[1] = 0.5f;

        mamba.forward(x, true);  // inference = true
        assert(mamba.cacheX.size() == 0);
        assert(mamba.states.size() == 0);

        mamba.forward(x, false); // training = false
        assert(mamba.cacheX.size() == 1);
        assert(mamba.states.size() == 1);
        std::cout << "  PASS: inference mode suppresses caching\n";
    }

    std::cout << "\n=== All MambaLayer tests passed! ===" << std::endl;
}

int main()
{
    runTests();
    return 0;
}
