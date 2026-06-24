#include "mamba.h"
#include <fstream>

namespace RL {

MambaLayer::MambaLayer(std::size_t inputDim_,
                       std::size_t hiddenDim_,
                       std::size_t outputDim_,
                       bool trainFlag)
    : inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    type = iLayer::LAYER_MAMBA;

    /* Input projection */
    bool useInputProj = (inputDim != hiddenDim);
    if (useInputProj) {
        W_in = Tensor(hiddenDim, inputDim);
        b_in = Tensor(hiddenDim, 1);
        Random::uniform(W_in, -0.1f, 0.1f);
        Random::uniform(b_in, -0.1f, 0.1f);
    }

    /*
     * Diagonal A: clamped to (0, 1] for stability.
     *   A_diag[i] = 1.0 - exp(-log(1 + exp(init))) 
     * We store raw logit values and clamp during application.
     * Initialized near 0.99 for long memory.
     */
    A_diag = Tensor(hiddenDim, 1);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        A_diag[i] = 0.9f + 0.09f * (float)std::rand() / RAND_MAX;
    }

    /* Selective B: B(x) = sigmoid(W_B · x + b_B) */
    W_B = Tensor(hiddenDim, inputDim);
    b_B = Tensor(hiddenDim, 1);
    Random::uniform(W_B, -0.1f, 0.1f);
    Random::uniform(b_B, -0.1f, 0.1f);

    /* Selective Δ: Δ(x) = softplus(W_Δ · x + b_Δ) */
    W_delta = Tensor(hiddenDim, inputDim);
    b_delta = Tensor(hiddenDim, 1);
    /* Initialize so that Δ starts small (~0.1) */
    Random::uniform(W_delta, -0.01f, 0.01f);
    Random::uniform(b_delta, 0.08f, 0.12f);

    /* Output projection */
    C = Tensor(outputDim, hiddenDim);
    b = Tensor(outputDim, 1);
    Random::uniform(C, -0.1f, 0.1f);
    Random::uniform(b, -0.1f, 0.1f);

    /* Persistent hidden state and output */
    h = Tensor(hiddenDim, 1);
    o = Tensor(outputDim, 1);

    if (trainFlag) {
        if (useInputProj) {
            g_W_in = Tensor(hiddenDim, inputDim);
            g_b_in = Tensor(hiddenDim, 1);
            v_W_in = Tensor(hiddenDim, inputDim);
            s_W_in = Tensor(hiddenDim, inputDim);
            v_b_in = Tensor(hiddenDim, 1);
            s_b_in = Tensor(hiddenDim, 1);
        }
        g_A_diag = Tensor(hiddenDim, 1);
        v_A_diag = Tensor(hiddenDim, 1);
        s_A_diag = Tensor(hiddenDim, 1);

        g_W_B = Tensor(hiddenDim, inputDim);
        g_b_B = Tensor(hiddenDim, 1);
        v_W_B = Tensor(hiddenDim, inputDim);
        s_W_B = Tensor(hiddenDim, inputDim);
        v_b_B = Tensor(hiddenDim, 1);
        s_b_B = Tensor(hiddenDim, 1);

        g_W_delta = Tensor(hiddenDim, inputDim);
        g_b_delta = Tensor(hiddenDim, 1);
        v_W_delta = Tensor(hiddenDim, inputDim);
        s_W_delta = Tensor(hiddenDim, inputDim);
        v_b_delta = Tensor(hiddenDim, 1);
        s_b_delta = Tensor(hiddenDim, 1);

        g_C = Tensor(outputDim, hiddenDim);
        g_b = Tensor(outputDim, 1);
        v_C = Tensor(outputDim, hiddenDim);
        s_C = Tensor(outputDim, hiddenDim);
        v_b = Tensor(outputDim, 1);
        s_b = Tensor(outputDim, 1);
    }
}

void MambaLayer::reset()
{
    h.zero();
    cacheX.clear();
    cacheE.clear();
    states.clear();
}

void MambaLayer::initParams()
{
    /* Re-init with same scheme as constructor */
    if (W_in.totalSize > 0) {
        Random::uniform(W_in, -0.1f, 0.1f);
        Random::uniform(b_in, -0.1f, 0.1f);
    }
    for (std::size_t i = 0; i < hiddenDim; i++) {
        A_diag[i] = 0.9f + 0.09f * (float)std::rand() / RAND_MAX;
    }
    Random::uniform(W_B, -0.1f, 0.1f);
    Random::uniform(b_B, -0.1f, 0.1f);
    Random::uniform(W_delta, -0.01f, 0.01f);
    Random::uniform(b_delta, 0.08f, 0.12f);
    Random::uniform(C, -0.1f, 0.1f);
    Random::uniform(b, -0.1f, 0.1f);
}

/*
 * MambaLayer::feedForward — single timestep with selective SSM
 *
 *   x_proj  = W_in · x + b_in              (if inputDim != hiddenDim, else x)
 *   B_sel   = sigmoid(W_B · x + b_B)       — input-dependent input gate
 *   Δ       = softplus(W_Δ · x + b_Δ)      — input-dependent step size
 *
 *   Ā       = exp(-Δ ⊙ (1 - A_diag))       — discretized decay (ZOH)
 *   B̄       = Δ ⊙ B_sel                    — discretized input
 *
 *   h(t)    = Ā ⊙ h(t-1) + B̄ ⊙ x_proj
 *   y(t)    = tanh(C · h(t) + b)
 */
MambaLayer::State MambaLayer::feedForward(const Tensor &x, const Tensor &_h)
{
    State state(hiddenDim, outputDim);

    /* ——— Input projection ——— */
    Tensor x_proj(hiddenDim, 1);
    if (W_in.totalSize > 0) {
        Tensor::MM::ikkj(x_proj, W_in, x);
        for (std::size_t i = 0; i < hiddenDim; i++) {
            x_proj[i] += b_in[i];
        }
    } else {
        /* inputDim == hiddenDim, just copy */
        for (std::size_t i = 0; i < hiddenDim; i++) {
            x_proj[i] = x[i];
        }
    }

    /* ——— Selective B(x) ——— */
    Tensor z_B(hiddenDim, 1);
    Tensor::MM::ikkj(z_B, W_B, x);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        state.B[i] = Sigmoid::f(z_B[i] + b_B[i]);
    }

    /* ——— Selective Δ(x) ——— */
    Tensor z_delta(hiddenDim, 1);
    Tensor::MM::ikkj(z_delta, W_delta, x);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float zd = z_delta[i] + b_delta[i];
        state.delta[i] = Softplus::f(zd);
    }

    /* ——— Discretization ——— */
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float A_i = (A_diag[i] > 1.0f) ? 1.0f : (A_diag[i] < 0.0f ? 0.0f : A_diag[i]);
        state.A_bar[i] = std::exp(-state.delta[i] * (1.0f - A_i));
        state.B_bar[i] = state.delta[i] * state.B[i];
    }

    /* ——— State update: h(t) = Ā ⊙ h(t-1) + B̄ ⊙ x_proj ——— */
    for (std::size_t i = 0; i < hiddenDim; i++) {
        state.h[i] = state.A_bar[i] * _h[i] + state.B_bar[i] * x_proj[i];
    }

    /* ——— Output: y = tanh(C · h + b) ——— */
    Tensor::MM::ikkj(state.y, C, state.h);
    for (std::size_t i = 0; i < outputDim; i++) {
        state.y[i] = Tanh::f(state.y[i] + b[i]);
    }

    return state;
}

Tensor &MambaLayer::forward(const Tensor &x, bool inference)
{
    State state = feedForward(x, h);
    h = state.h;
    o = state.y;

    if (!inference) {
        cacheX.push_back(x);
        states.push_back(state);
    }

    return o;
}

/*
 * MambaLayer::backwardAtTime — single timestep BPTT
 *
 * Given E = dL/dy(t), computes parameter gradients.
 *
 * Key derivative relationships:
 *   y = tanh(C·h + b)
 *   h(t) = Ā ⊙ h(t-1) + B̄ ⊙ x_proj
 *   Ā = exp(-Δ ⊙ (1 - A))
 *   B̄ = Δ ⊙ B
 *   B = sigmoid(W_B · x + b_B)
 *   Δ = softplus(W_Δ · x + b_Δ)
 */
void MambaLayer::backwardAtTime(int t,
                                const Tensor &x,
                                const Tensor &E,
                                State &delta_)
{
    /* ——— 1. Output error through tanh ——— */
    Tensor delta_y(outputDim, 1);
    for (std::size_t i = 0; i < outputDim; i++) {
        delta_y[i] = E[i] * Tanh::df(states[t].y[i]);
    }

    /* ——— 2. Output parameter gradients ——— */
    /* g.C += δy · h(t)^T */
    Tensor::MM::ikjk(g_C, delta_y, states[t].h);
    /* g.b += δy */
    g_b += delta_y;

    /* ——— 3. Hidden state gradient from output ——— */
    /* δh_from_output = C^T · δy */
    Tensor delta_h(hiddenDim, 1);
    Tensor::MM::kikj(delta_h, C, delta_y);

    /* Add future gradient propagated through Ā: δh += Ā ⊙ δh_next */
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float A_i = (A_diag[i] > 1.0f) ? 1.0f : (A_diag[i] < 0.0f ? 0.0f : A_diag[i]);
        float A_bar_i = std::exp(-states[t].delta[i] * (1.0f - A_i));
        delta_h[i] += A_bar_i * delta_.h[i];
    }

    /* ——— 4. Input projection gradient ——— */
    if (W_in.totalSize > 0) {
        /* x_proj contributes to h through: h += B̄ ⊙ x_proj
         * dL/dx_proj = delta_h ⊙ B̄
         * dL/dW_in = dL/dx_proj · x^T
         * dL/db_in = dL/dx_proj
         */
        Tensor dx_proj(hiddenDim, 1);
        for (std::size_t i = 0; i < hiddenDim; i++) {
            dx_proj[i] = delta_h[i] * states[t].B_bar[i];
        }
        Tensor::MM::ikjk(g_W_in, dx_proj, x);
        g_b_in += dx_proj;
    }

    /* ——— 5. Ā gradient and backprop to next timestep ——— */
    /* dĀ_grad = δh ⊙ h(t-1) */
    Tensor dA_bar(hiddenDim, 1);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        dA_bar[i] = delta_h[i] * (t > 0 ? states[t-1].h[i] : 0.0f);
    }

    /* B̄ gradient: g_B̄ = δh ⊙ x_proj */
    Tensor dB_bar(hiddenDim, 1);
    Tensor x_proj(hiddenDim, 1);
    if (W_in.totalSize > 0) {
        Tensor::MM::ikkj(x_proj, W_in, x);
        for (std::size_t i = 0; i < hiddenDim; i++) x_proj[i] += b_in[i];
    } else {
        for (std::size_t i = 0; i < hiddenDim; i++) x_proj[i] = x[i];
    }
    for (std::size_t i = 0; i < hiddenDim; i++) {
        dB_bar[i] = delta_h[i] * x_proj[i];
    }

    /* ——— 6. A_diag gradient ——— */
    /* Ā = exp(-Δ ⊙ (1 - A))
     * dĀ/dA = Ā ⊙ Δ
     */
    float A_bar_val, delta_val;
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float A_i = (A_diag[i] > 1.0f) ? 1.0f : (A_diag[i] < 0.0f ? 0.0f : A_diag[i]);
        A_bar_val = std::exp(-states[t].delta[i] * (1.0f - A_i));
        delta_val = states[t].delta[i];
        g_A_diag[i] += dA_bar[i] * A_bar_val * delta_val;
    }

    /* ——— 7. Δ gradient ——— */
    /* dĀ/dΔ = -Ā ⊙ (1 - A)
     * dB̄/dΔ = B (since B̄ = Δ ⊙ B)
     */
    Tensor dDelta(hiddenDim, 1);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float A_i = (A_diag[i] > 1.0f) ? 1.0f : (A_diag[i] < 0.0f ? 0.0f : A_diag[i]);
        A_bar_val = std::exp(-states[t].delta[i] * (1.0f - A_i));
        dDelta[i] = -dA_bar[i] * A_bar_val * (1.0f - A_i) + dB_bar[i] * states[t].B[i];
    }

    /* Δ = softplus(z_Δ) = log(1 + exp(z_Δ))
     * dΔ/dz_Δ = sigmoid(z_Δ)
     */
    Tensor z_delta(hiddenDim, 1);
    Tensor::MM::ikkj(z_delta, W_delta, x);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float sig = Sigmoid::f(z_delta[i] + b_delta[i]);
        float dDz = dDelta[i] * sig;
        g_W_delta(i, 0) += dDz * x[0];
        for (std::size_t j = 1; j < inputDim; j++) {
            g_W_delta(i, j) += dDz * x[j];
        }
        g_b_delta[i] += dDz;
    }

    /* ——— 8. B gradient (through sigmoid) ——— */
    /* B = sigmoid(z_B), dB̄/dB = Δ */
    Tensor dB_sel(hiddenDim, 1);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        dB_sel[i] = dB_bar[i] * states[t].delta[i];
    }

    /* B = sigmoid(W_B · x + b_B) — approximate */
    Tensor z_B(hiddenDim, 1);
    Tensor::MM::ikkj(z_B, W_B, x);
    for (std::size_t i = 0; i < hiddenDim; i++) {
        float sig = Sigmoid::f(z_B[i] + b_B[i]);
        float dBz = dB_sel[i] * Sigmoid::df(sig);
        g_W_B(i, 0) += dBz * x[0];
        for (std::size_t j = 1; j < inputDim; j++) {
            g_W_B(i, j) += dBz * x[j];
        }
        g_b_B[i] += dBz;
    }

    /* ——— 9. Propagate to previous timestep ——— */
    /* δh(t-1) = Ā ⊙ δh(t) */
    if (t > 0) {
        for (std::size_t i = 0; i < hiddenDim; i++) {
            delta_.h[i] = delta_h[i];
            /* multiply by Ā */
            float A_i = (A_diag[i] > 1.0f) ? 1.0f : (A_diag[i] < 0.0f ? 0.0f : A_diag[i]);
            delta_.h[i] *= std::exp(-states[t].delta[i] * (1.0f - A_i));
        }
    }
}

void MambaLayer::backward(const std::vector<Tensor> &x, const std::vector<Tensor> &E)
{
    State delta_(hiddenDim, outputDim);
    delta_.h.zero();

    for (int t = (int)states.size() - 1; t >= 0; t--) {
        backwardAtTime(t, x[t], E[t], delta_);
    }

    states.clear();
}

void MambaLayer::cacheError(const Tensor &e)
{
    cacheE.push_back(e);
}

/* ==================== Optimizers ==================== */

void MambaLayer::SGD(float lr)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    if (W_in.totalSize > 0) {
        Optimize::SGD(W_in, g_W_in, lr);
        Optimize::SGD(b_in, g_b_in, lr);
        g_W_in.zero();
        g_b_in.zero();
    }
    Optimize::SGD(A_diag, g_A_diag, lr);
    Optimize::SGD(W_B, g_W_B, lr);
    Optimize::SGD(b_B, g_b_B, lr);
    Optimize::SGD(W_delta, g_W_delta, lr);
    Optimize::SGD(b_delta, g_b_delta, lr);
    Optimize::SGD(C, g_C, lr);
    Optimize::SGD(b, g_b, lr);

    g_A_diag.zero();
    g_W_B.zero(); g_b_B.zero();
    g_W_delta.zero(); g_b_delta.zero();
    g_C.zero(); g_b.zero();
}

void MambaLayer::RMSProp(float lr, float rho, float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    if (W_in.totalSize > 0) {
        Optimize::RMSProp(W_in, v_W_in, g_W_in, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b_in, v_b_in, g_b_in, lr, rho, decay, clipGrad);
        g_W_in.zero(); g_b_in.zero();
    }
    Optimize::RMSProp(A_diag, v_A_diag, g_A_diag, lr, rho, decay, clipGrad);
    Optimize::RMSProp(W_B, v_W_B, g_W_B, lr, rho, decay, clipGrad);
    Optimize::RMSProp(b_B, v_b_B, g_b_B, lr, rho, decay, clipGrad);
    Optimize::RMSProp(W_delta, v_W_delta, g_W_delta, lr, rho, decay, clipGrad);
    Optimize::RMSProp(b_delta, v_b_delta, g_b_delta, lr, rho, decay, clipGrad);
    Optimize::RMSProp(C, v_C, g_C, lr, rho, decay, clipGrad);
    Optimize::RMSProp(b, v_b, g_b, lr, rho, decay, clipGrad);

    g_A_diag.zero();
    g_W_B.zero(); g_b_B.zero();
    g_W_delta.zero(); g_b_delta.zero();
    g_C.zero(); g_b.zero();
}

void MambaLayer::Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    if (W_in.totalSize > 0) {
        Optimize::Adam(W_in, v_W_in, s_W_in, g_W_in, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(b_in, v_b_in, s_b_in, g_b_in, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        g_W_in.zero(); g_b_in.zero();
    }
    Optimize::Adam(A_diag, v_A_diag, s_A_diag, g_A_diag, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(W_B, v_W_B, s_W_B, g_W_B, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(b_B, v_b_B, s_b_B, g_b_B, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(W_delta, v_W_delta, s_W_delta, g_W_delta, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(b_delta, v_b_delta, s_b_delta, g_b_delta, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(C, v_C, s_C, g_C, alpha_, beta_, lr, alpha, beta, decay, clipGrad);
    Optimize::Adam(b, v_b, s_b, g_b, alpha_, beta_, lr, alpha, beta, decay, clipGrad);

    g_A_diag.zero();
    g_W_B.zero(); g_b_B.zero();
    g_W_delta.zero(); g_b_delta.zero();
    g_C.zero(); g_b.zero();
}

void MambaLayer::clamp(float c0, float cn)
{
    if (W_in.totalSize > 0) {
        Optimize::clamp(W_in, c0, cn);
        Optimize::clamp(b_in, c0, cn);
    }
    Optimize::clamp(A_diag, 0.0f, 1.0f);
    Optimize::clamp(W_B, c0, cn);
    Optimize::clamp(b_B, c0, cn);
    Optimize::clamp(W_delta, c0, cn);
    Optimize::clamp(b_delta, c0, cn);
    Optimize::clamp(C, c0, cn);
    Optimize::clamp(b, c0, cn);
}

/* ==================== Parameter ops ==================== */

void MambaLayer::copyTo(iLayer *layer)
{
    MambaLayer &dst = *static_cast<MambaLayer*>(layer);
    dst.W_in = W_in;
    dst.b_in = b_in;
    dst.A_diag = A_diag;
    dst.W_B = W_B;
    dst.b_B = b_B;
    dst.W_delta = W_delta;
    dst.b_delta = b_delta;
    dst.C = C;
    dst.b = b;
}

void MambaLayer::softUpdateTo(iLayer *layer, float alpha)
{
    MambaLayer &dst = *static_cast<MambaLayer*>(layer);
    if (W_in.totalSize > 0) {
        lerp(dst.W_in, W_in, alpha);
        lerp(dst.b_in, b_in, alpha);
    }
    lerp(dst.A_diag, A_diag, alpha);
    lerp(dst.W_B, W_B, alpha);
    lerp(dst.b_B, b_B, alpha);
    lerp(dst.W_delta, W_delta, alpha);
    lerp(dst.b_delta, b_delta, alpha);
    lerp(dst.C, C, alpha);
    lerp(dst.b, b, alpha);
}

void MambaLayer::write(std::ofstream &file)
{
    /* Input projection */
    file << (W_in.totalSize > 0 ? W_in.toString() : std::string("0,0")) << std::endl;
    file << (b_in.totalSize > 0 ? b_in.toString() : std::string("0,0")) << std::endl;
    /* Core params */
    file << A_diag.toString() << std::endl;
    file << W_B.toString() << std::endl;
    file << b_B.toString() << std::endl;
    file << W_delta.toString() << std::endl;
    file << b_delta.toString() << std::endl;
    file << C.toString() << std::endl;
    file << b.toString() << std::endl;
}

void MambaLayer::read(std::ifstream &file)
{
    auto parse = [](std::ifstream &f) -> Tensor {
        std::string s;
        std::getline(f, s);
        if (s == "0,0") return Tensor();
        return Tensor::fromString(s);
    };
    W_in = parse(file);
    b_in = parse(file);
    A_diag = parse(file);
    W_B = parse(file);
    b_B = parse(file);
    W_delta = parse(file);
    b_delta = parse(file);
    C = parse(file);
    b = parse(file);
}

/* ==================== Test ==================== */

void MambaLayer::test()
{
    std::cout << "=== MambaLayer (Selective SSM) Test ===" << std::endl;

    MambaLayer mamba(2, 8, 1, true);

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
    std::cout << "Training MambaLayer on 40000 samples (mini-batches of 16)..." << std::endl;
    for (int i = 0; i < 1000; i++) {
        mamba.reset();
        for (int j = 0; j < 16; j++) {
            int k = selectIndex(Random::engine);
            Tensor &out = mamba.forward(data[k]);
            mamba.cacheError(Loss::MSE::df(out, target[k]));
        }
        mamba.RMSProp(0.01f, 0.9f, 0.0f, true);
    }

    /* Evaluation */
    mamba.reset();
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
            mamba.forward(p, true);
            float err = mamba.o[0] - z;
            totalError += std::abs(err);
            std::cout << "x=" << x << " y=" << y
                      << " target=" << z
                      << " pred=" << mamba.o[0]
                      << " err=" << err << std::endl;
        }
    }
    std::cout << "Average abs error: " << totalError / 100.0f << std::endl;
}

} // namespace RL
