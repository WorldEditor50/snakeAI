#include "../rl/ssm.h"
#include "../rl/loss.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace RL;

/*
 * Test SSM (State Space Model)
 *
 * Tests:
 * 1. Constructor and type registration
 * 2. Forward pass - single timestep
 * 3. Forward pass - sequential timesteps (state persistence)
 * 4. BPTT gradient flow
 * 5. MSE training on 1D function
 * 6. Reset
 * 7. Parameter operations (copyTo, softUpdateTo)
 * 8. Model save/load
 * 9. Inference mode
 */
void runTests()
{
    /* ==================== Test 1: Constructor & Type ==================== */
    {
        std::cout << "Test 1: SSM Construction & Type\n";
        SSM ssm(2, 8, 1, true);
        assert(ssm.type == iLayer::LAYER_SSM);
        assert(ssm.inputDim == 2);
        assert(ssm.hiddenDim == 8);
        assert(ssm.outputDim == 1);
        assert(ssm.A.totalSize == 8 * 8);
        assert(ssm.B.totalSize == 8 * 2);
        assert(ssm.C.totalSize == 1 * 8);
        assert(ssm.b.totalSize == 1);
        std::cout << "  PASS: dimensions correct\n";
    }

    /* ==================== Test 2: Single forward pass ==================== */
    {
        std::cout << "Test 2: Forward pass (single timestep)\n";
        SSM ssm(3, 16, 2, true);
        Tensor x(3, 1);
        x[0] = 0.5f; x[1] = -0.3f; x[2] = 0.8f;

        Tensor &out = ssm.forward(x);
        assert(out.totalSize == 2);
        // Output should be tanh-activated => values in [-1, 1]
        for (int i = 0; i < 2; i++) {
            assert(out[i] >= -1.0f && out[i] <= 1.0f);
        }
        assert(ssm.cacheX.size() == 1);
        assert(ssm.states.size() == 1);
        std::cout << "  PASS: forward produces valid tanh output\n";
    }

    /* ==================== Test 3: Sequential state persistence ==================== */
    {
        std::cout << "Test 3: Sequential forward (state persistence)\n";
        SSM ssm(1, 4, 1, true);

        Tensor x1(1, 1); x1[0] = 1.0f;
        Tensor x2(1, 1); x2[0] = 1.0f;

        ssm.forward(x1);
        Tensor first_h = ssm.h;
        ssm.forward(x2);
        Tensor second_h = ssm.h;

        // States should differ since A·h adds new info each step
        bool differs = false;
        for (int i = 0; i < 4; i++) {
            if (std::abs(second_h[i] - first_h[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        assert(differs);
        assert(ssm.cacheX.size() == 2);
        std::cout << "  PASS: hidden state evolves through time\n";
    }

    /* ==================== Test 4: BPTT gradient flow ==================== */
    {
        std::cout << "Test 4: BPTT gradient flow\n";
        SSM ssm(2, 6, 1, true);

        Tensor x1(2, 1); x1[0] = 0.1f; x1[1] = 0.2f;
        Tensor x2(2, 1); x2[0] = 0.3f; x2[1] = 0.4f;

        ssm.forward(x1);
        ssm.forward(x2);

        Tensor e1(1, 1); e1[0] = 1.0f;
        Tensor e2(1, 1); e2[0] = 1.0f;
        ssm.cacheError(e1);
        ssm.cacheError(e2);

        ssm.SGD(0.01f);

        // Gradients should be non-zero after BPTT
        bool gA_nz = false, gB_nz = false, gC_nz = false;
        for (int i = 0; i < 6 * 6; i++) if (std::abs(ssm.g.A[i]) > 1e-10f) gA_nz = true;
        for (int i = 0; i < 6 * 2; i++) if (std::abs(ssm.g.B[i]) > 1e-10f) gB_nz = true;
        for (int i = 0; i < 1 * 6; i++) if (std::abs(ssm.g.C[i]) > 1e-10f) gC_nz = true;

        assert(gA_nz && gB_nz && gC_nz);
        std::cout << "  PASS: BPTT computes non-zero gradients for A, B, C\n";
    }

    /* ==================== Test 5: Training on simple function ==================== */
    {
        std::cout << "Test 5: Training on z = sin(x^2 + y^2)\n";
        SSM ssm(2, 10, 1, true);

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
        ssm.reset();
        float initialLoss = 0;
        for (int i = 0; i < 30; i++) {
            int k = sel(Random::engine);
            ssm.forward(data[k], true);
            float d = ssm.o[0] - targets[k][0];
            initialLoss += d*d;
        }
        initialLoss /= 30;
        std::cout << "  Initial MSE: " << initialLoss << std::endl;

        // Training
        for (int iter = 0; iter < 200; iter++) {
            ssm.reset();
            for (int j = 0; j < 16; j++) {
                int k = sel(Random::engine);
                ssm.forward(data[k]);
                ssm.cacheError(Loss::MSE::df(ssm.o, targets[k]));
            }
            ssm.RMSProp(0.01f, 0.9f, 0.0f, true);
        }

        // Final loss
        ssm.reset();
        float finalLoss = 0;
        for (int i = 0; i < 30; i++) {
            int k = sel(Random::engine);
            ssm.forward(data[k], true);
            float d = ssm.o[0] - targets[k][0];
            finalLoss += d*d;
        }
        finalLoss /= 30;
        std::cout << "  Final MSE: " << finalLoss << std::endl;
        assert(finalLoss < initialLoss);
        std::cout << "  PASS: loss decreased after training\n";
    }

    /* ==================== Test 6: Reset ==================== */
    {
        std::cout << "Test 6: Reset\n";
        SSM ssm(2, 4, 1, true);
        Tensor x(2, 1); x[0] = 1.0f; x[1] = 2.0f;

        ssm.forward(x);
        ssm.forward(x);
        assert(ssm.cacheX.size() == 2);
        assert(ssm.states.size() == 2);

        ssm.reset();
        assert(ssm.cacheX.size() == 0);
        assert(ssm.states.size() == 0);
        for (int i = 0; i < 4; i++) assert(ssm.h[i] == 0.0f);
        std::cout << "  PASS: reset clears cache and state\n";
    }

    /* ==================== Test 7: copyTo and softUpdateTo ==================== */
    {
        std::cout << "Test 7: copyTo & softUpdateTo\n";
        SSM ssm1(2, 6, 1, true);
        SSM ssm2(2, 6, 1, false);

        ssm1.copyTo(&ssm2);
        for (int i = 0; i < 6 * 6; i++) assert(ssm2.A[i] == ssm1.A[i]);
        for (int i = 0; i < 6 * 2; i++) assert(ssm2.B[i] == ssm1.B[i]);
        for (int i = 0; i < 1 * 6; i++) assert(ssm2.C[i] == ssm1.C[i]);
        assert(ssm2.b[0] == ssm1.b[0]);

        // softUpdateTo with alpha=0.5: lerp(dst, src, alpha) => dst = alpha*dst + (1-alpha)*src
        float oldA0 = ssm2.A[0];
        ssm1.A.zero();
        ssm1.A[0] = 1.0f;
        ssm1.softUpdateTo(&ssm2, 0.5f);
        float expected = 0.5f * oldA0 + 0.5f * 0.0f;  // alpha*dst + (1-alpha)*src
        assert(std::abs(ssm2.A[0] - 0.5f * oldA0) < 1e-5f);
        std::cout << "  PASS: copyTo and softUpdateTo work correctly\n";
    }

    /* ==================== Test 8: Save/Load ==================== */
    {
        std::cout << "Test 8: Save/Load\n";
        SSM ssm1(2, 6, 1, true);
        SSM ssm2(2, 6, 1, false);

        ssm1.A.zero(); for (int i = 0; i < 6; i++) ssm1.A(i,i) = (float)(i+1);
        ssm1.B.zero(); for (int i = 0; i < 6*2; i++) ssm1.B[i] = (float)(i);
        ssm1.C.zero(); for (int i = 0; i < 6; i++) ssm1.C[i] = (float)(i+10);
        ssm1.b.zero(); ssm1.b[0] = 3.14f;

        std::ofstream ofs("ssm_test_save.txt");
        ssm1.write(ofs);
        ofs.close();

        std::ifstream ifs("ssm_test_save.txt");
        ssm2.read(ifs);
        ifs.close();

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                assert(std::abs(ssm2.A(i,j) - ssm1.A(i,j)) < 1e-5f);
            }
        }
        assert(std::abs(ssm2.b[0] - 3.14f) < 1e-5f);
        std::cout << "  PASS: save/load preserves all parameters\n";
    }

    /* ==================== Test 9: Inference mode ==================== */
    {
        std::cout << "Test 9: Inference mode (no caching)\n";
        SSM ssm(2, 6, 1, true);
        Tensor x(2, 1); x[0] = 0.5f; x[1] = 0.5f;

        ssm.forward(x, true);  // inference = true
        assert(ssm.cacheX.size() == 0);
        assert(ssm.states.size() == 0);

        ssm.forward(x, false); // training = false
        assert(ssm.cacheX.size() == 1);
        assert(ssm.states.size() == 1);
        std::cout << "  PASS: inference mode suppresses caching\n";
    }

    std::cout << "\n=== All SSM tests passed! ===" << std::endl;
}

int main()
{
    runTests();
    return 0;
}
