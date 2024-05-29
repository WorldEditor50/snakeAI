#include "lstm.h"

RL::LSTM::LSTM(std::size_t inputDim_,
               std::size_t hiddenDim_,
               std::size_t outputDim_,
               bool trainFlag):
    LSTMParam(inputDim_, hiddenDim_, outputDim_),
    inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    if (trainFlag == true) {
        d = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        v = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        s = LSTMParam(inputDim_, hiddenDim_, outputDim_);
    }
    h = Mat(hiddenDim, 1);
    c = Mat(hiddenDim, 1);
    y = Mat(outputDim, 1);
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

RL::LSTM::State RL::LSTM::feedForward(const RL::Mat &x, const RL::Mat &_h, const RL::Mat &_c)
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
    for (std::size_t i = 0; i < wi.rows; i++) {
        for (std::size_t j = 0; j < wi.cols; j++) {
            state.f[i] += wf(i, j) * x[j];
            state.i[i] += wi(i, j) * x[j];
            state.g[i] += wg(i, j) * x[j];
            state.o[i] += wo(i, j) * x[j];
        }
    }
    for (std::size_t i = 0; i < ui.rows; i++) {
        for (std::size_t j = 0; j < ui.cols; j++) {
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
    for (std::size_t i = 0; i < w.rows; i++) {
        for (std::size_t j = 0; j < w.cols; j++) {
            state.y[i] += w(i, j) * state.h[j];
        }
        state.y[i] = Linear::f(state.y[i] + b[i]);
    }
    return state;
}

void RL::LSTM::forward(const std::vector<RL::Mat> &sequence)
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

RL::Mat &RL::LSTM::forward(const RL::Mat &x)
{
    State state = feedForward(x, h, c);
    h = state.h;
    c = state.c;
    y = state.y;
    return y;
}

void RL::LSTM::backwardAtTime(int t,
                         const RL::Mat &x,
                         const RL::Mat &E,
                         State &delta_)
{
    State delta(hiddenDim, outputDim);
    for (std::size_t i = 0; i < w.rows; i++) {
        for (std::size_t j = 0; j < w.cols; j++) {
            delta.h[j] += w(i, j) * E[i];
        }
    }
    for (std::size_t i = 0; i < ui.rows; i++) {
        for (std::size_t j = 0; j < ui.cols; j++) {
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
    Mat f_ = t < states.size() - 1 ? states[t + 1].f : Mat(hiddenDim, 1);
    Mat _c = t > 0 ? states[t - 1].c : Mat(hiddenDim, 1);
    for (std::size_t i = 0; i < delta.o.size(); i++) {
        delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::d(states[t].c[i]) + delta_.c[i] * f_[i];
        delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::d(states[t].o[i]);
        delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::d(states[t].g[i]);
        delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::d(states[t].i[i]);
        delta.f[i] = delta.c[i] * _c[i] * Sigmoid::d(states[t].f[i]);
    }
    /* gradient */
    for (std::size_t i = 0; i < w.rows; i++) {
        for (std::size_t j = 0; j < w.cols; j++) {
            d.w(i, j) += E[i] * Linear::d(states[t].y[i]) * states[t].h[j];
        }
        d.b[i] += E[i] * Linear::d(states[t].y[i]);
    }
    for (std::size_t i = 0; i < wi.rows; i++) {
        for (std::size_t j = 0; j < wi.cols; j++) {
            d.wi(i, j) += delta.i[i] * x[j];
            d.wf(i, j) += delta.f[i] * x[j];
            d.wg(i, j) += delta.g[i] * x[j];
            d.wo(i, j) += delta.o[i] * x[j];
        }
    }
    Mat _h = t > 0 ? states[t - 1].h : Mat(hiddenDim, 1);
    for (std::size_t i = 0; i < ui.rows; i++) {
        for (std::size_t j = 0; j < ui.cols; j++) {
            d.ui(i, j) += delta.i[i] * _h[j];
            d.uf(i, j) += delta.f[i] * _h[j];
            d.ug(i, j) += delta.g[i] * _h[j];
            d.uo(i, j) += delta.o[i] * _h[j];
        }
    }
    for (std::size_t i = 0; i < bi.size(); i++) {
        d.bi[i] += delta.i[i];
        d.bf[i] += delta.f[i];
        d.bg[i] += delta.g[i];
        d.bo[i] += delta.o[i];
    }
    /* next */
    delta_ = delta;
    return;
}

void RL::LSTM::backward(const std::vector<RL::Mat> &x, const std::vector<RL::Mat> &E)
{
    State delta_(hiddenDim, outputDim);
    /* backward through time */
    for (int t = states.size() - 1; t >= 0; t--) {
        backwardAtTime(t, x[t], E[t], delta_);
    }
    states.clear();
    return;
}

void RL::LSTM::gradient(const std::vector<RL::Mat> &x,
                        const std::vector<RL::Mat> &yt)
{
    /* loss */
    std::vector<RL::Mat> E(states.size(), Mat(outputDim, 1));
    for (int t = states.size() - 1; t >= 0; t--) {
        for (std::size_t i = 0; i < outputDim; i++) {
            E[t][i] = 2* (states[t].y[i] - yt[t][i]);
        }
    }
    /* backward */
    backward(x, E);
    return;
}

void RL::LSTM::gradient(const std::vector<RL::Mat> &x, const RL::Mat &yt)
{
    /* loss */
    std::vector<RL::Mat> E(states.size(), Mat(outputDim, 1));
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
    Optimize::SGD(w, d.w, learningRate);
    Optimize::SGD(b, d.b, learningRate);

    Optimize::SGD(wi, d.wi, learningRate);
    Optimize::SGD(wg, d.wg, learningRate);
    Optimize::SGD(wf, d.wf, learningRate);
    Optimize::SGD(wo, d.wo, learningRate);

    Optimize::SGD(ui, d.ui, learningRate);
    Optimize::SGD(ug, d.ug, learningRate);
    Optimize::SGD(uf, d.uf, learningRate);
    Optimize::SGD(uo, d.uo, learningRate);

    Optimize::SGD(bi, d.bi, learningRate);
    Optimize::SGD(bg, d.bg, learningRate);
    Optimize::SGD(bf, d.bf, learningRate);
    Optimize::SGD(bo, d.bo, learningRate);
    d.zero();
    return;
}

void RL::LSTM::RMSProp(float learningRate, float rho, float decay)
{
    Optimize::RMSProp(w, s.w, d.w, learningRate, rho, decay);
    Optimize::RMSProp(b, s.b, d.b, learningRate, rho, decay);

    Optimize::RMSProp(wi, s.wi, d.wi, learningRate, rho, decay);
    Optimize::RMSProp(wg, s.wg, d.wg, learningRate, rho, decay);
    Optimize::RMSProp(wf, s.wf, d.wf, learningRate, rho, decay);
    Optimize::RMSProp(wo, s.wo, d.wo, learningRate, rho, decay);

    Optimize::RMSProp(ui, s.ui, d.ui, learningRate, rho, decay);
    Optimize::RMSProp(ug, s.ug, d.ug, learningRate, rho, decay);
    Optimize::RMSProp(uf, s.uf, d.uf, learningRate, rho, decay);
    Optimize::RMSProp(uo, s.uo, d.uo, learningRate, rho, decay);

    Optimize::RMSProp(bi, s.bi, d.bi, learningRate, rho, decay);
    Optimize::RMSProp(bg, s.bg, d.bg, learningRate, rho, decay);
    Optimize::RMSProp(bf, s.bf, d.bf, learningRate, rho, decay);
    Optimize::RMSProp(bo, s.bo, d.bo, learningRate, rho, decay);

    d.zero();
    return;
}

void RL::LSTM::Adam(float learningRate,  float alpha, float beta, float decay)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimize::Adam(w, s.w, v.w, d.w, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(b, s.b, v.b, d.b, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(wi, s.wi, v.wi, d.wi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wg, s.wg, v.wg, d.wg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wf, s.wf, v.wf, d.wf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(wo, s.wo, v.wo, d.wo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(ui, s.ui, v.ui, d.ui, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(ug, s.ug, v.ug, d.ug, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(uf, s.uf, v.uf, d.uf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(uo, s.uo, v.uo, d.uo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(bi, s.bi, v.bi, d.bi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bg, s.bg, v.bg, d.bg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bf, s.bf, v.bf, d.bf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(bo, s.bo, v.bo, d.bo, alpha_t, beta_t, learningRate, alpha, beta, decay);
    d.zero();
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
    RL::EMA(dst.wi, wi, rho);
    RL::EMA(dst.wg, wg, rho);
    RL::EMA(dst.wf, wf, rho);
    RL::EMA(dst.wo, wo, rho);
    RL::EMA(dst.ui, ui, rho);
    RL::EMA(dst.ug, ug, rho);
    RL::EMA(dst.uf, uf, rho);
    RL::EMA(dst.uo, uo, rho);
    RL::EMA(dst.bi, bi, rho);
    RL::EMA(dst.bg, bg, rho);
    RL::EMA(dst.bf, bf, rho);
    RL::EMA(dst.bo, bo, rho);
    RL::EMA(dst.w, w, rho);
    RL::EMA(dst.b, b, rho);
    return;
}

void RL::LSTM::test()
{
    LSTM lstm(2, 8, 1, true);
    auto zeta = [](float x, float y) -> float {
        return std::sin(x*x + y*y);
    };
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Mat> data;
    std::vector<Mat> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Mat p(2, 1);
            float z = zeta(i, j);
            p[0] = i;
            p[1] = j;
            Mat q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Mat> &batchData,
            std::vector<Mat> &batchTarget, int batchSize){
        int k = selectIndex(Rand::engine);
        while (k > data.size() - batchSize) {
            k = selectIndex(Rand::engine);
        }
        for (int i = 0; i < batchSize; i++) {
            batchData.push_back(data[k + i]);
            batchTarget.push_back(target[k + i]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Mat> batchData;
        std::vector<Mat> batchTarget;
        sample(batchData, batchTarget, 32);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.Adam(1e-3);
    }

    lstm.reset();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Mat p(2, 1);
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
