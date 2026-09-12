/*
 * test_gradcheck.cpp — targeted numerical-gradient (finite-difference) checks.
 *
 * Every check compares an ANALYTIC gradient produced by the library's own
 * backward() against a CENTRAL finite difference of the same loss, so a wrong
 * gradient shows up as a mismatch independent of whether training "looks" fine.
 *
 * These cover the specific defects that were fixed:
 *   [1]  Layer<Fn>::forward must not accumulate into `o` across calls
 *   [2]  Layer<Fn> backward gradient (forward-buffer handling)
 *   [3]  Net::backward input gradient  dL/dx   (used by VAE's decoder)
 *   [4]  ScaledConcat backward (softmax-Jacobian chain + sublayer errors)
 *   [5]  MultiHeadAttention backward (no spurious `ei += Wo^T e`)
 *   [6]  MOE gating buffer must not accumulate across forwards
 *   [7]  SSM forward recurrence h(t) = A*h(t-1) + B*x(t)  (was 2*A*h)
 *   [8]  SSM BPTT gradient                              (was A^T twice)
 *   [9]  MambaLayer BPTT gradient                       (was A_bar twice)
 *   [10] Conv2d kernel gradient AND input gradient (tanh' factor)
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include "rl/layer.h"
#include "rl/net.hpp"
#include "rl/loss.h"
#include "rl/concat.hpp"
#include "rl/attention.hpp"
#include "rl/moe.hpp"
#include "rl/conv2d.hpp"
#include "rl/ssm.h"
#include "rl/mamba.h"

using namespace std;
using RL::Tensor;
using RL::Layer;
using RL::Tanh;
using RL::Sigmoid;
using RL::Softmax;
namespace Loss = RL::Loss;      /* Loss is a NAMESPACE, not a class */
using RL::Net;

static int g_pass = 0;
static int g_fail = 0;

static void report(bool ok, const string &name, const string &detail)
{
    if (ok) {
        ++g_pass;
        cout << "  PASS  " << name;
    } else {
        ++g_fail;
        cout << "  FAIL  " << name;
    }
    if (!detail.empty()) {
        cout << "   [" << detail << "]";
    }
    cout << endl;
}

/* Relative mismatch between an analytic gradient and a central finite
   difference, with an ABSOLUTE noise floor.
 *
 * The loss is float32, so a central difference cannot resolve a gradient
 * component below roughly machine_eps*|loss|/(2*eps) ~ 1e-4. Without a floor,
 * a near-zero component (e.g. the input gradient at a padding-adjacent pixel)
 * produces an absolute error of ~1e-5 against a true value of ~1e-4, which a
 * pure ratio reports as a ~10% "failure" even though the two agree to within
 * the precision the method can offer. Anything within the floor is a match. */
static float relDiff(float num, float ana)
{
    float absErr = std::fabs(num - ana);
    if (absErr <= 1e-4f) {
        return 0.0f;
    }
    float denom = std::fabs(num) + std::fabs(ana) + 1e-4f;
    return absErr/denom;
}

static Tensor mk(std::initializer_list<float> v)
{
    Tensor t(static_cast<int>(v.size()), 1);
    int i = 0;
    for (float x : v) {
        t[i++] = x;
    }
    return t;
}

static float sumSqDiff(const Tensor &o, const Tensor &t)
{
    float s = 0;
    for (std::size_t i = 0; i < t.totalSize; i++) {
        float d = o[i] - t[i];
        s += d*d;
    }
    return s;
}

/* ============================================================ [1] & [2] */
static void test_layer_forward_and_grad()
{
    cout << "\n[1] Layer<Tanh>::forward must not accumulate into `o`" << endl;
    {
        Layer<Tanh> L(4, 3, true, true);
        Tensor xa = mk({0.5f, -1.0f, 2.0f, 0.25f});
        Tensor xb = mk({-2.0f, 0.75f, -0.5f, 1.5f});
        L.forward(xa);
        Tensor ref = L.o;                 // deep copy: correct output for xa
        L.forward(xb);
        L.forward(xa);                    // must reproduce `ref` exactly
        float worst = 0;
        for (std::size_t i = 0; i < ref.totalSize; i++) {
            worst = std::max(worst, std::fabs(ref[i] - L.o[i]));
        }
        report(worst < 1e-6f,
               "forward(xa) reproducible after an intervening forward(xb)",
               "maxdiff=" + to_string(worst));
    }
    {
        /* Same property, but with a fresh layer forwarded twice back-to-back:
           with accumulation the second output would be ~2x the first. */
        Layer<Tanh> L(3, 2, true, true);
        Tensor x = mk({0.3f, -0.4f, 0.8f});
        L.forward(x);
        Tensor first = L.o;
        L.forward(x);
        float worst = 0;
        for (std::size_t i = 0; i < first.totalSize; i++) {
            worst = std::max(worst, std::fabs(first[i] - L.o[i]));
        }
        report(worst < 1e-6f, "two identical forwards give identical output",
               "maxdiff=" + to_string(worst));
    }

    cout << "\n[2] Layer<Tanh> backward gradient vs finite difference" << endl;
    {
        Layer<Tanh> L(4, 3, true, true);
        Tensor x = mk({0.4f, -0.9f, 1.3f, 0.2f});
        Tensor t = mk({0.5f, -0.25f, 0.75f});

        auto loss = [&]() {
            L.forward(x);
            return sumSqDiff(L.o, t);
        };

        L.g.zero();
        L.forward(x);
        /* backward() reads the layer's OWN error member `e`; its argument is
           the OUTPUT buffer that receives dL/dx. */
        L.e = Loss::MSE::df(L.o, t);
        Tensor ei(L.inputDim, 1);
        ei.zero();
        L.backward(x, ei);

        const float eps = 1e-3f;
        float worstW = 0, worstB = 0;
        for (int i = 0; i < L.w.shape[0]; i++) {
            for (int j = 0; j < L.w.shape[1]; j++) {
                float orig = L.w(i, j);
                L.w(i, j) = orig + eps;
                float lp = loss();
                L.w(i, j) = orig - eps;
                float lm = loss();
                L.w(i, j) = orig;
                worstW = std::max(worstW, relDiff((lp - lm)/(2*eps), L.g.w(i, j)));
            }
        }
        for (int i = 0; i < L.b.totalSize; i++) {
            float orig = L.b[i];
            L.b[i] = orig + eps;
            float lp = loss();
            L.b[i] = orig - eps;
            float lm = loss();
            L.b[i] = orig;
            worstB = std::max(worstB, relDiff((lp - lm)/(2*eps), L.g.b[i]));
        }
        report(worstW < 1e-2f, "dL/dw matches finite difference",
               "worst rel err=" + to_string(worstW));
        report(worstB < 1e-2f, "dL/db matches finite difference",
               "worst rel err=" + to_string(worstB));

        /* dL/dx must also match (this is what the previous layer consumes) */
        float worstX = 0;
        for (int k = 0; k < x.totalSize; k++) {
            float orig = x[k];
            x[k] = orig + eps;
            float lp = loss();
            x[k] = orig - eps;
            float lm = loss();
            x[k] = orig;
            worstX = std::max(worstX, relDiff((lp - lm)/(2*eps), ei[k]));
        }
        report(worstX < 1e-2f, "dL/dx (input gradient) matches finite difference",
               "worst rel err=" + to_string(worstX));
    }
}

/* ================================================================= [3] */
static void test_net_input_gradient()
{
    cout << "\n[3] Net::backward input gradient dL/dx (VAE decoder relies on this)" << endl;
    {
        Net net(Layer<Tanh>::_(3, 5, true, true),
                Layer<Tanh>::_(5, 4, true, true),
                Layer<Softmax>::_(4, 2, true, true));
        Tensor x = mk({0.6f, -1.1f, 0.35f});
        Tensor t = mk({0.8f, 0.2f});

        auto loss = [&]() {
            Tensor &o = net.forward(x);
            return sumSqDiff(o, t);
        };

        Tensor o = net.forward(x);
        Tensor e = Loss::MSE::df(o, t);
        net.backward(x, e);
        Tensor anaIn = net.inputGrad;      // dL/dx captured by Net::backward

        report(anaIn.totalSize == static_cast<std::size_t>(x.totalSize),
               "inputGrad is sized like the network INPUT (not layer[0]'s output)",
               "size=" + to_string(anaIn.totalSize) + " expected=" + to_string(x.totalSize));

        const float eps = 1e-3f;
        float worst = 0;
        float norm = 0;
        for (int k = 0; k < x.totalSize; k++) {
            float orig = x[k];
            x[k] = orig + eps;
            float lp = loss();
            x[k] = orig - eps;
            float lm = loss();
            x[k] = orig;
            worst = std::max(worst, relDiff((lp - lm)/(2*eps), anaIn[k]));
            norm += std::fabs(anaIn[k]);
        }
        report(norm > 1e-6f && worst < 1e-2f,
               "dL/dx matches finite difference",
               "worst rel err=" + to_string(worst) + " |grad|=" + to_string(norm));
    }
}

/* ================================================================= [4] */
static void test_scaledconcat_grad()
{
    cout << "\n[4] ScaledConcat backward vs finite difference" << endl;
    {
        RL::ScaledConcat<Layer<Sigmoid>, 3> sc(Layer<Sigmoid>(2, 2, true, true),
                                              2, 2, true);
        const int outDim = sc.outputDim;          // unitDim*N = 6
        Tensor x = mk({0.7f, -0.3f});
        Tensor t(outDim, 1);
        for (int i = 0; i < outDim; i++) {
            t[i] = 0.1f*float(i) - 0.25f;
        }

        auto loss = [&]() {
            Tensor &o = sc.forward(x);
            return sumSqDiff(o, t);
        };

        sc.g.zero();
        Tensor &o = sc.forward(x);
        sc.e = Loss::MSE::df(o, t);
        Tensor ei(2, 1);
        ei.zero();
        sc.backward(x, ei);

        const float eps = 1e-3f;
        float worstW1 = 0, worstW2 = 0, worstB = 0, worstSub = 0;

        for (int i = 0; i < sc.w1.shape[0]; i++) {
            for (int j = 0; j < sc.w1.shape[1]; j++) {
                float orig = sc.w1(i, j);
                sc.w1(i, j) = orig + eps; float lp = loss();
                sc.w1(i, j) = orig - eps; float lm = loss();
                sc.w1(i, j) = orig;
                worstW1 = std::max(worstW1, relDiff((lp - lm)/(2*eps), sc.g.w1(i, j)));
            }
        }
        for (int i = 0; i < sc.w2.shape[0]; i++) {
            for (int j = 0; j < sc.w2.shape[1]; j++) {
                float orig = sc.w2(i, j);
                sc.w2(i, j) = orig + eps; float lp = loss();
                sc.w2(i, j) = orig - eps; float lm = loss();
                sc.w2(i, j) = orig;
                worstW2 = std::max(worstW2, relDiff((lp - lm)/(2*eps), sc.g.w2(i, j)));
            }
        }
        for (int i = 0; i < sc.b.totalSize; i++) {
            float orig = sc.b[i];
            sc.b[i] = orig + eps; float lp = loss();
            sc.b[i] = orig - eps; float lm = loss();
            sc.b[i] = orig;
            worstB = std::max(worstB, relDiff((lp - lm)/(2*eps), sc.g.b[i]));
        }
        /* one representative sub-layer weight: exercises the softmax-Jacobian
           slicing of the sub-layer error, which used to be a raw slice of e */
        for (int j = 0; j < sc.layers[0].w.shape[1]; j++) {
            float orig = sc.layers[0].w(0, j);
            sc.layers[0].w(0, j) = orig + eps; float lp = loss();
            sc.layers[0].w(0, j) = orig - eps; float lm = loss();
            sc.layers[0].w(0, j) = orig;
            worstSub = std::max(worstSub, relDiff((lp - lm)/(2*eps), sc.layers[0].g.w(0, j)));
        }

        report(worstW1 < 2e-2f, "dL/dw1 matches finite difference", "worst=" + to_string(worstW1));
        report(worstW2 < 2e-2f, "dL/dw2 matches finite difference", "worst=" + to_string(worstW2));
        report(worstB  < 2e-2f, "dL/db  matches finite difference", "worst=" + to_string(worstB));
        report(worstSub < 2e-2f, "sub-layer dL/dw matches finite difference",
               "worst=" + to_string(worstSub));

        /* x-path: ScaledConcat also feeds x through w2, so ei must match */
        float worstX = 0;
        for (int k = 0; k < x.totalSize; k++) {
            float orig = x[k];
            x[k] = orig + eps; float lp = loss();
            x[k] = orig - eps; float lm = loss();
            x[k] = orig;
            worstX = std::max(worstX, relDiff((lp - lm)/(2*eps), ei[k]));
        }
        report(worstX < 2e-2f, "dL/dx matches finite difference", "worst=" + to_string(worstX));
    }
}

/* ================================================================= [5] */
static void test_mha_grad()
{
    cout << "\n[5] MultiHeadAttention backward vs finite difference" << endl;
    {
        RL::MultiHeadAttention<2> mha(4, 4, true);   // 2 heads, d_k = 2
        Tensor x = mk({0.5f, -0.8f, 1.2f, 0.3f});
        Tensor t(4, 1);
        for (int i = 0; i < 4; i++) {
            t[i] = 0.3f - 0.15f*float(i);
        }

        auto loss = [&]() {
            Tensor &o = mha.forward(x);
            return sumSqDiff(o, t);
        };

        mha.g.zero();
        Tensor &o = mha.forward(x);
        mha.e = Loss::MSE::df(o, t);
        Tensor ei(4, 1);
        ei.zero();
        mha.backward(x, ei);

        const float eps = 1e-3f;
        float worstWo = 0, worstX = 0;
        for (int i = 0; i < mha.wo.shape[0]; i++) {
            for (int j = 0; j < mha.wo.shape[1]; j++) {
                float orig = mha.wo(i, j);
                mha.wo(i, j) = orig + eps; float lp = loss();
                mha.wo(i, j) = orig - eps; float lm = loss();
                mha.wo(i, j) = orig;
                worstWo = std::max(worstWo, relDiff((lp - lm)/(2*eps), mha.g.wo(i, j)));
            }
        }
        for (int k = 0; k < x.totalSize; k++) {
            float orig = x[k];
            x[k] = orig + eps; float lp = loss();
            x[k] = orig - eps; float lm = loss();
            x[k] = orig;
            worstX = std::max(worstX, relDiff((lp - lm)/(2*eps), ei[k]));
        }
        report(worstWo < 2e-2f, "dL/dWo matches finite difference", "worst=" + to_string(worstWo));
        report(worstX  < 5e-2f, "dL/dx  matches finite difference (no extra Wo^T term)",
               "worst=" + to_string(worstX));
    }
}

/* ================================================================= [6] */
static void test_moe_forward_repeatable()
{
    cout << "\n[6] MOE gating buffer must not accumulate across forwards" << endl;
    {
        RL::MOE<3, 2> moe(4, true);              // 3 experts, d_model = 4
        Tensor xa = mk({0.6f, -0.2f, 0.9f, -0.7f});
        Tensor xb = mk({-1.0f, 0.4f, 0.1f, 0.8f});

        moe.forward(xa);
        Tensor ref = moe.o;
        Tensor gateRef = moe.gate;
        moe.forward(xb);
        moe.forward(xa);
        float worstO = 0, worstGate = 0;
        for (std::size_t i = 0; i < ref.totalSize; i++) {
            worstO = std::max(worstO, std::fabs(ref[i] - moe.o[i]));
        }
        for (std::size_t i = 0; i < gateRef.totalSize; i++) {
            worstGate = std::max(worstGate, std::fabs(gateRef[i] - moe.gate[i]));
        }
        report(worstO < 1e-6f, "MOE output reproducible after an intervening forward",
               "maxdiff=" + to_string(worstO));
        report(worstGate < 1e-6f, "MOE gate reproducible (was accumulating)",
               "maxdiff=" + to_string(worstGate));
        float gsum = 0;
        for (std::size_t i = 0; i < moe.gate.totalSize; i++) {
            gsum += moe.gate[i];
        }
        report(std::fabs(gsum - 1.0f) < 1e-4f, "MOE gate is still a probability distribution",
               "sum=" + to_string(gsum));
    }
}

/* ================================================================= [7] */
static void test_ssm_forward_recurrence()
{
    cout << "\n[7] SSM forward recurrence  h(t) = A*h(t-1) + B*x(t)" << endl;
    {
        RL::SSM ssm(2, 3, 3, true);
        Tensor x1 = mk({0.5f, -0.25f});
        Tensor x2 = mk({-0.8f, 0.6f});

        ssm.reset();
        ssm.forward(x1);
        Tensor h1 = ssm.h;                      // deep copy of h(1)
        ssm.forward(x2);
        Tensor h2 = ssm.h;

        Tensor expect(3, 1);
        Tensor::MM::ikkj(expect, ssm.A, h1);    // A*h(1)
        Tensor::MM::ikkj(expect, ssm.B, x2);    // + B*x(2)
        float worst = 0;
        for (std::size_t i = 0; i < expect.totalSize; i++) {
            worst = std::max(worst, std::fabs(expect[i] - h2[i]));
        }
        report(worst < 1e-5f, "h(2) == A*h(1) + B*x(2)  (not 2*A*h + B*x)",
               "maxdiff=" + to_string(worst));
    }
}

/* ================================================================= [8] */
static void test_ssm_bptt_grad()
{
    cout << "\n[8] SSM BPTT gradient vs finite difference (3-step sequence)" << endl;
    {
        const int inDim = 2, hidDim = 3, outDim = 2, steps = 3;
        RL::SSM ssm(inDim, hidDim, outDim, true);

        vector<Tensor> xs;
        xs.push_back(mk({0.4f, -0.6f}));
        xs.push_back(mk({0.9f, 0.2f}));
        xs.push_back(mk({-0.3f, 0.7f}));
        vector<Tensor> ts;
        ts.push_back(mk({0.25f, -0.5f}));
        ts.push_back(mk({-0.1f, 0.6f}));
        ts.push_back(mk({0.4f, 0.3f}));

        auto loss = [&]() {
            ssm.reset();
            float s = 0;
            for (int t = 0; t < steps; t++) {
                Tensor &o = ssm.forward(xs[t]);
                s += sumSqDiff(o, ts[t]);
            }
            return s;
        };

        ssm.g.zero();
        ssm.reset();
        for (int t = 0; t < steps; t++) {
            Tensor &o = ssm.forward(xs[t]);
            ssm.cacheError(Loss::MSE::df(o, ts[t]));
        }
        ssm.backward(ssm.cacheX, ssm.cacheE);

        Tensor anaA = ssm.g.A;
        Tensor anaB = ssm.g.B;
        Tensor anaC = ssm.g.C;

        const float eps = 1e-3f;
        float worstA = 0, worstB = 0, worstC = 0;
        for (int i = 0; i < ssm.A.shape[0]; i++) {
            for (int j = 0; j < ssm.A.shape[1]; j++) {
                float orig = ssm.A(i, j);
                ssm.A(i, j) = orig + eps; float lp = loss();
                ssm.A(i, j) = orig - eps; float lm = loss();
                ssm.A(i, j) = orig;
                worstA = std::max(worstA, relDiff((lp - lm)/(2*eps), anaA(i, j)));
            }
        }
        for (int i = 0; i < ssm.B.shape[0]; i++) {
            for (int j = 0; j < ssm.B.shape[1]; j++) {
                float orig = ssm.B(i, j);
                ssm.B(i, j) = orig + eps; float lp = loss();
                ssm.B(i, j) = orig - eps; float lm = loss();
                ssm.B(i, j) = orig;
                worstB = std::max(worstB, relDiff((lp - lm)/(2*eps), anaB(i, j)));
            }
        }
        for (int i = 0; i < ssm.C.shape[0]; i++) {
            for (int j = 0; j < ssm.C.shape[1]; j++) {
                float orig = ssm.C(i, j);
                ssm.C(i, j) = orig + eps; float lp = loss();
                ssm.C(i, j) = orig - eps; float lm = loss();
                ssm.C(i, j) = orig;
                worstC = std::max(worstC, relDiff((lp - lm)/(2*eps), anaC(i, j)));
            }
        }
        report(worstA < 2e-2f, "dL/dA matches finite difference", "worst=" + to_string(worstA));
        report(worstB < 2e-2f, "dL/dB matches finite difference", "worst=" + to_string(worstB));
        report(worstC < 2e-2f, "dL/dC matches finite difference", "worst=" + to_string(worstC));
    }
}

/* ================================================================= [9] */
static void test_mamba_bptt_grad()
{
    cout << "\n[9] MambaLayer BPTT gradient vs finite difference (3-step sequence)" << endl;
    {
        const int inDim = 2, hidDim = 3, outDim = 2, steps = 3;
        RL::MambaLayer mb(inDim, hidDim, outDim, true);

        vector<Tensor> xs;
        xs.push_back(mk({0.5f, -0.4f}));
        xs.push_back(mk({-0.2f, 0.8f}));
        xs.push_back(mk({0.6f, 0.1f}));
        vector<Tensor> ts;
        ts.push_back(mk({0.3f, -0.2f}));
        ts.push_back(mk({-0.4f, 0.5f}));
        ts.push_back(mk({0.2f, 0.7f}));

        auto loss = [&]() {
            mb.reset();
            float s = 0;
            for (int t = 0; t < steps; t++) {
                Tensor &o = mb.forward(xs[t]);
                s += sumSqDiff(o, ts[t]);
            }
            return s;
        };

        mb.g_C.zero();
        mb.g_b.zero();
        mb.g_A_diag.zero();
        mb.reset();
        for (int t = 0; t < steps; t++) {
            Tensor &o = mb.forward(xs[t]);
            mb.cacheError(Loss::MSE::df(o, ts[t]));
        }
        mb.backward(mb.cacheX, mb.cacheE);

        Tensor anaC = mb.g_C;
        Tensor anaA = mb.g_A_diag;

        const float eps = 1e-3f;
        float worstC = 0, worstA = 0;
        for (int i = 0; i < mb.C.shape[0]; i++) {
            for (int j = 0; j < mb.C.shape[1]; j++) {
                float orig = mb.C(i, j);
                mb.C(i, j) = orig + eps; float lp = loss();
                mb.C(i, j) = orig - eps; float lm = loss();
                mb.C(i, j) = orig;
                worstC = std::max(worstC, relDiff((lp - lm)/(2*eps), anaC(i, j)));
            }
        }
        /* A_diag enters the loss only INDIRECTLY (through the decay factor),
           so its gradient is small and a 1e-3 central difference suffers from
           float32 cancellation noise. Use a larger step and print the pairs. */
        const float epsA = 2e-2f;
        for (int i = 0; i < mb.A_diag.totalSize; i++) {
            float orig = mb.A_diag[i];
            mb.A_diag[i] = orig + epsA; float lp = loss();
            mb.A_diag[i] = orig - epsA; float lm = loss();
            mb.A_diag[i] = orig;
            float num = (lp - lm)/(2*epsA);
            float rd = relDiff(num, anaA[i]);
            cout << "        A_diag[" << i << "]  analytic=" << setprecision(6) << anaA[i]
                 << "  numeric=" << num << "  rel=" << rd << endl;
            worstA = std::max(worstA, rd);
        }
        report(worstC < 2e-2f, "dL/dC matches finite difference", "worst=" + to_string(worstC));
        report(worstA < 5e-2f, "dL/dA_diag matches finite difference (single A_bar)",
               "worst=" + to_string(worstA));
    }
}

/* ================================================================ [10] */
static void test_conv2d_grad()
{
    cout << "\n[10] Conv2d kernel + input gradient vs finite difference" << endl;
    {
        /* 1 input channel, 4x4 image, 1 output channel, 2x2 kernel, stride 1 */
        RL::Conv2d<Tanh> conv(1, 4, 4, 1, 2, 1, 0, true, true);
        Tensor x(1, 4, 4);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                x(0, i, j) = 0.3f*float(i) - 0.2f*float(j) + 0.1f;
            }
        }
        Tensor target(conv.o.totalSize, 1);
        for (std::size_t i = 0; i < target.totalSize; i++) {
            target[i] = 0.2f*float(i) - 0.3f;
        }

        auto loss = [&]() {
            Tensor &o = conv.forward(x);
            return sumSqDiff(o, target);
        };

        conv.g.zero();
        Tensor &o = conv.forward(x);
        conv.e = Loss::MSE::df(o, target);
        Tensor ei(1, 4, 4);
        ei.zero();
        conv.backward(x, ei);

        Tensor anaK = conv.g.kernels;
        Tensor anaEi = ei;

        const float eps = 1e-3f;
        float worstK = 0, worstX = 0;
        for (int oc = 0; oc < conv.kernels.shape[0]; oc++) {
            for (int ic = 0; ic < conv.kernels.shape[1]; ic++) {
                for (int u = 0; u < conv.kernels.shape[2]; u++) {
                    for (int v = 0; v < conv.kernels.shape[3]; v++) {
                        float orig = conv.kernels(oc, ic, u, v);
                        conv.kernels(oc, ic, u, v) = orig + eps; float lp = loss();
                        conv.kernels(oc, ic, u, v) = orig - eps; float lm = loss();
                        conv.kernels(oc, ic, u, v) = orig;
                        worstK = std::max(worstK,
                            relDiff((lp - lm)/(2*eps), anaK(oc, ic, u, v)));
                    }
                }
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float orig = x(0, i, j);
                x(0, i, j) = orig + eps; float lp = loss();
                x(0, i, j) = orig - eps; float lm = loss();
                x(0, i, j) = orig;
                worstX = std::max(worstX, relDiff((lp - lm)/(2*eps), anaEi(0, i, j)));
            }
        }
        report(worstK < 2e-2f, "dL/dkernel matches finite difference",
               "worst=" + to_string(worstK));
        report(worstX < 2e-2f, "dL/dx matches finite difference (tanh' included)",
               "worst=" + to_string(worstX));
    }
}

/* =================================================================== main */
int main()
{
    /* Deterministic: the library seeds its generators from std::random_device
       at static-init time, which would make these checks vary run to run. */
    std::srand(12345);
    RL::Random::engine.seed(12345);
    RL::Random::generator.seed(12345);

    cout << "============================================================" << endl;
    cout << "Targeted gradient verification (analytic vs finite difference)" << endl;
    cout << "============================================================" << endl;

    test_layer_forward_and_grad();
    test_net_input_gradient();
    test_scaledconcat_grad();
    test_mha_grad();
    test_moe_forward_repeatable();
    test_ssm_forward_recurrence();
    test_ssm_bptt_grad();
    test_mamba_bptt_grad();
    test_conv2d_grad();

    cout << "\n============================================================" << endl;
    cout << "Summary: " << g_pass << " passed, " << g_fail << " failed" << endl;
    cout << "============================================================" << endl;
    return g_fail;
}
