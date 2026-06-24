#include "ssm.h"
#include <fstream>

namespace RL {

SSM::SSM(std::size_t inputDim_,
         std::size_t hiddenDim_,
         std::size_t outputDim_,
         bool trainFlag)
    : inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    type = iLayer::LAYER_SSM;

    /* Allocate parameters */
    A = Tensor(hiddenDim, hiddenDim);
    B = Tensor(hiddenDim, inputDim);
    C = Tensor(outputDim, hiddenDim);
    b = Tensor(outputDim, 1);

    /*
     * Initialize A as a near-identity matrix for stable dynamics:
     *   A[i,i] ~ 0.99 (stable eigenvalues)
     *   A[i,j] small random for j != i
     */
    for (std::size_t i = 0; i < hiddenDim; i++) {
        for (std::size_t j = 0; j < hiddenDim; j++) {
            if (i == j) {
                A(i, j) = 0.95f + 0.04f * (float)std::rand() / RAND_MAX;
            } else {
                A(i, j) = 0.02f * (2.0f * (float)std::rand() / RAND_MAX - 1.0f);
            }
        }
    }

    /* Initialize B, C, b with small random values */
    for (std::size_t i = 0; i < B.totalSize; i++) B[i] = 0.1f * (2.0f * (float)std::rand() / RAND_MAX - 1.0f);
    for (std::size_t i = 0; i < C.totalSize; i++) C[i] = 0.1f * (2.0f * (float)std::rand() / RAND_MAX - 1.0f);
    for (std::size_t i = 0; i < b.totalSize; i++) b[i] = 0.0f;

    /* Persistent hidden state */
    h = Tensor(hiddenDim, 1);
    o = Tensor(outputDim, 1);

    if (trainFlag == true) {
        g = SSMParam(inputDim_, hiddenDim_, outputDim_);
        v = SSMParam(inputDim_, hiddenDim_, outputDim_);
        s = SSMParam(inputDim_, hiddenDim_, outputDim_);
    }
}

void SSM::reset()
{
    h.zero();
    cacheX.clear();
    cacheE.clear();
    states.clear();
    return;
}

void SSM::initParams()
{
    /* Re-init with same scheme as constructor */
    for (std::size_t i = 0; i < hiddenDim; i++) {
        for (std::size_t j = 0; j < hiddenDim; j++) {
            if (i == j) {
                A(i, j) = 0.95f + 0.04f * (float)std::rand() / RAND_MAX;
            } else {
                A(i, j) = 0.02f * (2.0f * (float)std::rand() / RAND_MAX - 1.0f);
            }
        }
    }
    Random::uniform(B, -0.1f, 0.1f);
    Random::uniform(C, -0.1f, 0.1f);
    b.zero();
    return;
}

/*
 * SSM::feedForward — single timestep
 *
 *   h(t) = A · h(t-1) + B · x(t)
 *   y(t) = tanh(C · h(t) + b)
 */
SSM::State SSM::feedForward(const Tensor &x, const Tensor &_h)
{
    State state(hiddenDim, outputDim);

    /* h(t) = A · h(t-1) + B · x(t) */
    Tensor::MM::ikkj(state.h, A, _h);
    Tensor ah = state.h;  // save A·h for addition
    Tensor::MM::ikkj(state.h, B, x);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        state.h[i] += ah[i];
    }

    /* y(t) = tanh(C · h(t) + b) */
    Tensor::MM::ikkj(state.y, C, state.h);
    for (std::size_t i = 0; i < outputDim; i++) {
        state.y[i] = Tanh::f(state.y[i] + b[i]);
    }

    return state;
}

Tensor &SSM::forward(const Tensor &x, bool inference)
{
    State state = feedForward(x, h);
    h = state.h;
    o = state.y;

    if (inference == false) {
        cacheX.push_back(x);
        states.push_back(state);
    }

    return o;
}

/*
 * SSM::backwardAtTime — single timestep BPTT
 *
 * Forward computation:
 *   h(t) = A·h(t-1) + B·x(t)
 *   y(t) = tanh(C·h(t) + b)
 *
 * Given e = dL/dy(t):
 *   δy(t) = e ⊙ tanh'(y(t))
 *   dC   += δy(t) · h(t)^T
 *   db   += δy(t)
 *   δh(t)_from_output = C^T · δy(t)
 *
 * Total δh(t) = δh(t)_from_output + δh(t+1)_from_state (via A)
 *
 * Then:
 *   δA   += δh(t) · h(t-1)^T
 *   δB   += δh(t) · x(t)^T
 *   δh(t-1) = A^T · δh(t)   (propagate to previous timestep)
 */
void SSM::backwardAtTime(int t,
                         const Tensor &x,
                         const Tensor &E,
                         State &delta_)
{
    State delta(hiddenDim, outputDim);

    /* δy = E ⊙ tanh'(y) — output error through tanh activation */
    for (std::size_t i = 0; i < outputDim; i++) {
        delta.y[i] = E[i] * Tanh::df(states[t].y[i]);
    }

    /* ∂L/∂h += C^T · δy  (output-to-hidden contribution) */
    Tensor::MM::kikj(delta.h, C, delta.y);

    /* Add future gradient propagated through A:  ∂L/∂h += A^T · δh_{t+1} */
    Tensor::MM::kikj(delta.h, A, delta_.h);

    /* Parameter gradients */
    Tensor::MM::ikjk(g.C, delta.y, states[t].h);  // g.C += δy · h(t)^T
    g.b += delta.y;                                // g.b  += δy

    Tensor::MM::ikjk(g.A, delta.h, states[t>0 ? t-1 : 0].h);  // g.A += δh · h(t-1)^T
    // ^ Note: for t=0, we use h(-1) which was initial zero  (states[0] is zeroed)

    Tensor::MM::ikjk(g.B, delta.h, x);  // g.B += δh · x(t)^T

    /* Propagate gradient to previous timestep: δh(t-1) = A^T · δh(t) */
    if (t > 0) {
        Tensor::MM::kikj(delta_.h, A, delta.h);
    }

    return;
}

/*
 * SSM::backward — Backpropagation Through Time for full sequence
 *
 * Iterates from last timestep to first, calling backwardAtTime for each.
 */
void SSM::backward(const std::vector<Tensor> &x, const std::vector<Tensor> &E)
{
    State delta_(hiddenDim, outputDim);
    delta_.h.zero();

    for (int t = (int)states.size() - 1; t >= 0; t--) {
        backwardAtTime(t, x[t], E[t], delta_);
    }

    states.clear();
    return;
}

void SSM::cacheError(const Tensor &e)
{
    cacheE.push_back(e);
    return;
}

/* ==================== Optimizers ==================== */

void SSM::SGD(float lr)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    Optimize::SGD(A, g.A, lr);
    Optimize::SGD(B, g.B, lr);
    Optimize::SGD(C, g.C, lr);
    Optimize::SGD(b, g.b, lr);

    g.zero();
    return;
}

void SSM::RMSProp(float lr, float rho, float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    Optimize::RMSProp(A, v.A, g.A, lr, rho, decay, clipGrad);
    Optimize::RMSProp(B, v.B, g.B, lr, rho, decay, clipGrad);
    Optimize::RMSProp(C, v.C, g.C, lr, rho, decay, clipGrad);
    Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);

    g.zero();
    return;
}

void SSM::Adam(float lr, float alpha, float beta,
               float alpha_, float beta_,
               float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    Optimize::Adam(A, v.A, s.A, g.A, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(B, v.B, s.B, g.B, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(C, v.C, s.C, g.C, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(b, v.b, s.b, g.b, alpha_, beta_, lr, alpha, beta, decay, clipGrad);

    g.zero();
    return;
}

void SSM::clamp(float c0, float cn)
{
    Optimize::clamp(A, c0, cn);
    Optimize::clamp(B, c0, cn);
    Optimize::clamp(C, c0, cn);
    Optimize::clamp(b, c0, cn);
    return;
}

/* ==================== Parameter ops ==================== */

void SSM::copyTo(iLayer *layer)
{
    SSM &dst = *static_cast<SSM*>(layer);
    dst.A = A;
    dst.B = B;
    dst.C = C;
    dst.b = b;
    return;
}

void SSM::softUpdateTo(iLayer *layer, float alpha)
{
    SSM &dst = *static_cast<SSM*>(layer);
    lerp(dst.A, A, alpha);
    lerp(dst.B, B, alpha);
    lerp(dst.C, C, alpha);
    lerp(dst.b, b, alpha);
    return;
}

void SSM::write(std::ofstream &file)
{
    file << A.toString() << std::endl;
    file << B.toString() << std::endl;
    file << C.toString() << std::endl;
    file << b.toString() << std::endl;
    return;
}

void SSM::read(std::ifstream &file)
{
    std::string s;
    std::getline(file, s); A = Tensor::fromString(s);
    std::getline(file, s); B = Tensor::fromString(s);
    std::getline(file, s); C = Tensor::fromString(s);
    std::getline(file, s); b = Tensor::fromString(s);
    return;
}

/* ==================== Test ==================== */

void SSM::test()
{
    std::cout << "=== SSM (State Space Model) Test ===" << std::endl;

    SSM ssm(2, 8, 1, true);

    /* Generate training data: z = sin(x*x + y*y) */
    auto zeta = [](float x, float y) -> float {
        return std::sin(x*x + y*y);
    };

    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Tensor> data;
    std::vector<Tensor> target;

    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Tensor q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }

    std::uniform_int_distribution<int> selectIndex(0, (int)data.size() - 1);

    /* Training loop */
    std::cout << "Training SSM on 40000 samples (mini-batches of 16)..." << std::endl;
    for (int i = 0; i < 1000; i++) {
        ssm.reset();
        for (int j = 0; j < 16; j++) {
            int k = selectIndex(Random::engine);
            Tensor &out = ssm.forward(data[k]);
            ssm.cacheError(Loss::MSE::df(out, target[k]));
        }
        ssm.RMSProp(0.01f, 0.9f, 0.0f, true);
    }

    /* Evaluation */
    ssm.reset();
    std::cout << "\nEvaluation:" << std::endl;
    float totalError = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Tensor p(2, 1);
            float x = uniform(Random::engine);
            float y = uniform(Random::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            ssm.forward(p, true);
            float err = ssm.o[0] - z;
            totalError += std::abs(err);
            std::cout << "x=" << x << " y=" << y
                      << " target=" << z
                      << " pred=" << ssm.o[0]
                      << " err=" << err << std::endl;
        }
    }
    std::cout << "Average abs error: " << totalError / 100.0f << std::endl;
    return;
}

} // namespace RL
