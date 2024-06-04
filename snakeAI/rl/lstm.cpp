#include "lstm.h"

RL::LSTM::LSTM(std::size_t inputDim_,
               std::size_t hiddenDim_,
               std::size_t outputDim_,
               bool trainFlag):
    LSTMParam(inputDim_, hiddenDim_, outputDim_),
    inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    if (trainFlag == true) {
        g = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        v = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        s = LSTMParam(inputDim_, hiddenDim_, outputDim_);
    }
    h = Tensor(hiddenDim, 1);
    c = Tensor(hiddenDim, 1);
    y = Tensor(outputDim, 1);
    alpha_t = 1;
    beta_t = 1;
    LSTMParam::random();
}

void RL::LSTM::reset()
{
    h.zero();
    c.zero();
    return;
}

RL::LSTM::State RL::LSTM::feedForward(const RL::Tensor &x, const RL::Tensor &_h, const RL::Tensor &_c)
{
    /*
                                                         y
                                                         |
                                                        h(t)
                                      c(t)               |
        c(t-1) -->--x-----------------+----------------------->-- c(t)
                    |                 |             |    |
                    |                 |            tanh  |
                    |                 |             |    |
                    |          -------x      -------x-----
                 f  |        i |      | g    | o    |
                    |          |      |      |      |
                 sigmoid    sigmoid  tanh  sigmoid  |
                    |          |      |      |      |
        h(t-1) -->----------------------------      --------->--- h(t)
                    |
                    x(t)

        ft = sigmoid(Wf*xt + Uf*ht-1 + bf);
        it = sigmoid(Wi*xt + Ui*ht-1 + bi);
        gt = tanh(Wg*xt + Ug*ht-1 + bg);
        ot = sigmoid(Wo*xt + Uo*ht-1 + bo);
        ct = ft ⊙ ct-1 + it ⊙ gt
        ht = ot ⊙ tanh(ct)
        yt = linear(W*ht + b)
    */
    State state(hiddenDim, outputDim);
    for (std::size_t i = 0; i < wi.shape[0]; i++) {
        for (std::size_t j = 0; j < wi.shape[1]; j++) {
            state.f[i] += wf(i, j) * x[j];
            state.i[i] += wi(i, j) * x[j];
            state.g[i] += wg(i, j) * x[j];
            state.o[i] += wo(i, j) * x[j];
        }
    }
    for (std::size_t i = 0; i < ui.shape[0]; i++) {
        for (std::size_t j = 0; j < ui.shape[1]; j++) {
            state.f[i] += uf(i, j) * _h[j];
            state.i[i] += ui(i, j) * _h[j];
            state.g[i] += ug(i, j) * _h[j];
            state.o[i] += uo(i, j) * _h[j];
        }
    }
    for (std::size_t i = 0; i < state.f.size(); i++) {
        state.f[i] = Sigmoid::f(state.f[i] + bf[i]);
        state.i[i] = Sigmoid::f(state.i[i] + bi[i]);
        state.g[i] =    Tanh::f(state.g[i] + bg[i]);
        state.o[i] = Sigmoid::f(state.o[i] + bo[i]);
        state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
        state.h[i] = state.o[i] * Tanh::f(state.c[i]);
    }
    //Tensor::Mul::ikkj(state.y, w, state.h);
    //state.y += b;
    for (std::size_t i = 0; i < w.shape[0]; i++) {
        for (std::size_t j = 0; j < w.shape[1]; j++) {
            state.y[i] += w(i, j) * state.h[j];
        }
        state.y[i] = Linear::f(state.y[i] + b[i]);
    }
    return state;
}

void RL::LSTM::forward(const std::vector<RL::Tensor> &sequence)
{
    h.zero();
    c.zero();
    for (auto &x : sequence) {
        State state = feedForward(x, h, c);
        h = state.h;
        c = state.c;
        states.push_back(state);
    }
    return;
}

RL::Tensor &RL::LSTM::forward(const RL::Tensor &x)
{
    State state = feedForward(x, h, c);
    h = state.h;
    c = state.c;
    y = state.y;
    return y;
}

void RL::LSTM::backwardAtTime(int t,
                         const RL::Tensor &x,
                         const RL::Tensor &E,
                         State &delta_)
{
    State delta(hiddenDim, outputDim);
    for (std::size_t i = 0; i < w.shape[0]; i++) {
        for (std::size_t j = 0; j < w.shape[1]; j++) {
            delta.h[j] += w(i, j) * E[i];
        }
    }
    for (std::size_t i = 0; i < ui.shape[0]; i++) {
        for (std::size_t j = 0; j < ui.shape[1]; j++) {
            delta.h[j] += ui(i, j) * delta_.i[i];
            delta.h[j] += uf(i, j) * delta_.f[i];
            delta.h[j] += ug(i, j) * delta_.g[i];
            delta.h[j] += uo(i, j) * delta_.o[i];
        }
    }

    /*
        δht = E + δht+1
        δct = δht ⊙ ot ⊙ dtanh(ct) + δct+1 ⊙ ft+1
        δot = δht ⊙ tanh(ct) ⊙ dsigmoid(ot)
        δgt = δct ⊙ it ⊙ dtanh(gt)
        δit = δct ⊙ gt ⊙ dsigmoid(it)
        δft = δct ⊙ ct-1 ⊙ dsigmoid(ft)
    */
    Tensor f_ = t < states.size() - 1 ? states[t + 1].f : Tensor(hiddenDim, 1);
    Tensor _c = t > 0 ? states[t - 1].c : Tensor(hiddenDim, 1);
    for (std::size_t i = 0; i < delta.o.size(); i++) {
        delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::d(states[t].c[i]) + delta_.c[i] * f_[i];
        delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::d(states[t].o[i]);
        delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::d(states[t].g[i]);
        delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::d(states[t].i[i]);
        delta.f[i] = delta.c[i] * _c[i] * Sigmoid::d(states[t].f[i]);
    }
    /* gradient */
    for (std::size_t i = 0; i < w.shape[0]; i++) {
        for (std::size_t j = 0; j < w.shape[1]; j++) {
            g.w(i, j) += E[i] * Linear::d(states[t].y[i]) * states[t].h[j];
        }
        g.b[i] += E[i] * Linear::d(states[t].y[i]);
    }
    for (std::size_t i = 0; i < wi.shape[0]; i++) {
        for (std::size_t j = 0; j < wi.shape[1]; j++) {
            g.wi(i, j) += delta.i[i] * x[j];
            g.wf(i, j) += delta.f[i] * x[j];
            g.wg(i, j) += delta.g[i] * x[j];
            g.wo(i, j) += delta.o[i] * x[j];
        }
    }
    Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
    for (std::size_t i = 0; i < ui.shape[0]; i++) {
        for (std::size_t j = 0; j < ui.shape[1]; j++) {
            g.ui(i, j) += delta.i[i] * _h[j];
            g.uf(i, j) += delta.f[i] * _h[j];
            g.ug(i, j) += delta.g[i] * _h[j];
            g.uo(i, j) += delta.o[i] * _h[j];
        }
    }
    for (std::size_t i = 0; i < bi.size(); i++) {
        g.bi[i] += delta.i[i];
        g.bf[i] += delta.f[i];
        g.bg[i] += delta.g[i];
        g.bo[i] += delta.o[i];
    }
    /* next */
    delta_ = delta;
    return;
}

void RL::LSTM::backward(const std::vector<RL::Tensor> &x, const std::vector<RL::Tensor> &E)
{
    State delta_(hiddenDim, outputDim);
    /* backward through time */
    for (int t = states.size() - 1; t >= 0; t--) {
        backwardAtTime(t, x[t], E[t], delta_);
    }
    states.clear();
    return;
}

void RL::LSTM::gradient(const std::vector<RL::Tensor> &x,
                        const std::vector<RL::Tensor> &yt)
{
    /* loss */
    std::vector<RL::Tensor> E(states.size(), Tensor(outputDim, 1));
    for (int t = states.size() - 1; t >= 0; t--) {
        for (std::size_t i = 0; i < outputDim; i++) {
            E[t][i] = 2* (states[t].y[i] - yt[t][i]);
        }
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::LSTM::gradient(const std::vector<RL::Tensor> &x, const RL::Tensor &yt)
{
    /* loss */
    std::vector<RL::Tensor> E(states.size(), Tensor(outputDim, 1));
    int t = states.size() - 1;
    for (std::size_t i = 0; i < outputDim; i++) {
        E[t][i] = 2 * (states[t].y[i] - yt[i]);
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::LSTM::SGD(float learningRate)
{   
    Optimize::SGD(w, g.w, learningRate);
    Optimize::SGD(b, g.b, learningRate);

    Optimize::SGD(wi, g.wi, learningRate);
    Optimize::SGD(wg, g.wg, learningRate);
    Optimize::SGD(wf, g.wf, learningRate);
    Optimize::SGD(wo, g.wo, learningRate);

    Optimize::SGD(ui, g.ui, learningRate);
    Optimize::SGD(ug, g.ug, learningRate);
    Optimize::SGD(uf, g.uf, learningRate);
    Optimize::SGD(uo, g.uo, learningRate);

    Optimize::SGD(bi, g.bi, learningRate);
    Optimize::SGD(bg, g.bg, learningRate);
    Optimize::SGD(bf, g.bf, learningRate);
    Optimize::SGD(bo, g.bo, learningRate);
    g.zero();
    return;
}

void RL::LSTM::RMSProp(float learningRate, float rho, float decay)
{
    Optimize::RMSProp(w, s.w, g.w, learningRate, rho, decay);
    Optimize::RMSProp(b, s.b, g.b, learningRate, rho, decay);

    Optimize::RMSProp(wi, s.wi, g.wi, learningRate, rho, decay);
    Optimize::RMSProp(wg, s.wg, g.wg, learningRate, rho, decay);
    Optimize::RMSProp(wf, s.wf, g.wf, learningRate, rho, decay);
    Optimize::RMSProp(wo, s.wo, g.wo, learningRate, rho, decay);

    Optimize::RMSProp(ui, s.ui, g.ui, learningRate, rho, decay);
    Optimize::RMSProp(ug, s.ug, g.ug, learningRate, rho, decay);
    Optimize::RMSProp(uf, s.uf, g.uf, learningRate, rho, decay);
    Optimize::RMSProp(uo, s.uo, g.uo, learningRate, rho, decay);

    Optimize::RMSProp(bi, s.bi, g.bi, learningRate, rho, decay);
    Optimize::RMSProp(bg, s.bg, g.bg, learningRate, rho, decay);
    Optimize::RMSProp(bf, s.bf, g.bf, learningRate, rho, decay);
    Optimize::RMSProp(bo, s.bo, g.bo, learningRate, rho, decay);

    g.zero();
    return;
}

void RL::LSTM::Adam(float learningRate,  float alpha, float beta, float decay)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimize::Adam(w, s.w, v.w, g.w, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(b, s.b, v.b, g.b, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(wi, s.wi, v.wi, g.wi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wg, s.wg, v.wg, g.wg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wf, s.wf, v.wf, g.wf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wo, s.wo, v.wo, g.wo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(ui, s.ui, v.ui, g.ui, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(ug, s.ug, v.ug, g.ug, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(uf, s.uf, v.uf, g.uf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(uo, s.uo, v.uo, g.uo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(bi, s.bi, v.bi, g.bi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bg, s.bg, v.bg, g.bg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bf, s.bf, v.bf, g.bf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bo, s.bo, v.bo, g.bo, alpha_t, beta_t, learningRate, alpha, beta, decay);
    g.zero();
    return;
}

void RL::LSTM::clamp(float c0, float cn)
{
    Optimize::clamp(w, c0, cn);
    Optimize::clamp(b, c0, cn);

    Optimize::clamp(wi, c0, cn);
    Optimize::clamp(wg, c0, cn);
    Optimize::clamp(wf, c0, cn);
    Optimize::clamp(wo, c0, cn);

    Optimize::clamp(ui, c0, cn);
    Optimize::clamp(ug, c0, cn);
    Optimize::clamp(uf, c0, cn);
    Optimize::clamp(uo, c0, cn);

    Optimize::clamp(bi, c0, cn);
    Optimize::clamp(bg, c0, cn);
    Optimize::clamp(bf, c0, cn);
    Optimize::clamp(bo, c0, cn);
    return;
}

void RL::LSTM::copyTo(LSTM &dst)
{
    dst.wi = wi;
    dst.wg = wg;
    dst.wf = wf;
    dst.wo = wo;

    dst.ui = ui;
    dst.ug = ug;
    dst.uf = uf;
    dst.uo = uo;

    dst.bi = bi;
    dst.bg = bg;
    dst.bf = bf;
    dst.bo = bo;

    dst.w = w;
    dst.b = b;
    return;
}

void RL::LSTM::softUpdateTo(LSTM &dst, float rho)
{
    RL::lerp(dst.wi, wi, rho);
    RL::lerp(dst.wg, wg, rho);
    RL::lerp(dst.wf, wf, rho);
    RL::lerp(dst.wo, wo, rho);
    RL::lerp(dst.ui, ui, rho);
    RL::lerp(dst.ug, ug, rho);
    RL::lerp(dst.uf, uf, rho);
    RL::lerp(dst.uo, uo, rho);
    RL::lerp(dst.bi, bi, rho);
    RL::lerp(dst.bg, bg, rho);
    RL::lerp(dst.bf, bf, rho);
    RL::lerp(dst.bo, bo, rho);
    RL::lerp(dst.w, w, rho);
    RL::lerp(dst.b, b, rho);
    return;
}

void RL::LSTM::test()
{
    LSTM lstm(2, 8, 1, true);
    auto zeta = [](float x, float y) -> float {
        return std::sin(x*x + y*y);
    };
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Tensor> data;
    std::vector<Tensor> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Tensor p(2, 1);
            float z = zeta(i, j);
            p[0] = i;
            p[1] = j;
            Tensor q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Tensor> &batchData,
            std::vector<Tensor> &batchTarget, int batchSize){
        int k = selectIndex(Random::engine);
        while (k > data.size() - batchSize) {
            k = selectIndex(Random::engine);
        }
        for (int i = 0; i < batchSize; i++) {
            batchData.push_back(data[k + i]);
            batchTarget.push_back(target[k + i]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Tensor> batchData;
        std::vector<Tensor> batchTarget;
        sample(batchData, batchTarget, 32);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.Adam(1e-3);
    }

    lstm.reset();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Tensor p(2, 1);
            float z = zeta(i, j);
            p[0] = i;
            p[1] = j;
            auto s = lstm.forward(p);
            std::cout<<"x = "<<i<<" y = "<<j<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
