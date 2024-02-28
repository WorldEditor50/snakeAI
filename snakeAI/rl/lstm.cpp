#include "lstm.h"

RL::LSTM::LSTM(std::size_t inputDim_,
               std::size_t hiddenDim_,
               std::size_t outputDim_,
               bool trainFlag):
    LSTMParam(inputDim_, hiddenDim_, outputDim_),
    ema(false),gamma(0.9),
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
    for (std::size_t i = 0; i < Wi.rows; i++) {
        for (std::size_t j = 0; j < Wi.cols; j++) {
            state.f[i] += Wf(i, j) * x[j];
            state.i[i] += Wi(i, j) * x[j];
            state.g[i] += Wg(i, j) * x[j];
            state.o[i] += Wo(i, j) * x[j];
        }
    }
    for (std::size_t i = 0; i < Ui.rows; i++) {
        for (std::size_t j = 0; j < Ui.cols; j++) {
            state.f[i] += Uf(i, j) * _h[j];
            state.i[i] += Ui(i, j) * _h[j];
            state.g[i] += Ug(i, j) * _h[j];
            state.o[i] += Uo(i, j) * _h[j];
        }
    }
    for (std::size_t i = 0; i < state.f.size(); i++) {
        state.f[i] = Sigmoid::f(state.f[i] + Bf[i]);
        state.i[i] = Sigmoid::f(state.i[i] + Bi[i]);
        state.g[i] =    Tanh::f(state.g[i] + Bg[i]);
        state.o[i] = Sigmoid::f(state.o[i] + Bo[i]);
        state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
        state.h[i] = state.o[i] * Tanh::f(state.c[i]);
    }
    for (std::size_t i = 0; i < W.rows; i++) {
        for (std::size_t j = 0; j < W.cols; j++) {
            state.y[i] += W(i, j) * state.h[j];
        }
        state.y[i] = Linear::f(state.y[i] + B[i]);
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
    for (std::size_t i = 0; i < W.rows; i++) {
        for (std::size_t j = 0; j < W.cols; j++) {
            delta.h[j] += W(i, j) * E[i];
        }
    }
    if (ema == false) {
        for (std::size_t i = 0; i < Ui.rows; i++) {
            for (std::size_t j = 0; j < Ui.cols; j++) {
                delta.h[j] += Ui(i, j) * delta_.i[i];
                delta.h[j] += Uf(i, j) * delta_.f[i];
                delta.h[j] += Ug(i, j) * delta_.g[i];
                delta.h[j] += Uo(i, j) * delta_.o[i];
            }
        }
    } else {
        /* BPTT with EMA */
        for (std::size_t i = 0; i < Ui.rows; i++) {
            for (std::size_t j = 0; j < Ui.cols; j++) {
                delta.h[j] = delta.h[j]*gamma + Ui(i, j) * delta_.i[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Uf(i, j) * delta_.f[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Ug(i, j) * delta_.g[i]*(1 - gamma);
                delta.h[j] = delta.h[j]*gamma + Uo(i, j) * delta_.o[i]*(1 - gamma);
            }
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
    for (std::size_t i = 0; i < W.rows; i++) {
        for (std::size_t j = 0; j < W.cols; j++) {
            d.W(i, j) += E[i] * Linear::d(states[t].y[i]) * states[t].h[j];
        }
        d.B[i] += E[i] * Linear::d(states[t].y[i]);
    }
    for (std::size_t i = 0; i < Wi.rows; i++) {
        for (std::size_t j = 0; j < Wi.cols; j++) {
            d.Wi(i, j) += delta.i[i] * x[j];
            d.Wf(i, j) += delta.f[i] * x[j];
            d.Wg(i, j) += delta.g[i] * x[j];
            d.Wo(i, j) += delta.o[i] * x[j];
        }
    }
    Mat _h = t > 0 ? states[t - 1].h : Mat(hiddenDim, 1);
    for (std::size_t i = 0; i < Ui.rows; i++) {
        for (std::size_t j = 0; j < Ui.cols; j++) {
            d.Ui(i, j) += delta.i[i] * _h[j];
            d.Uf(i, j) += delta.f[i] * _h[j];
            d.Ug(i, j) += delta.g[i] * _h[j];
            d.Uo(i, j) += delta.o[i] * _h[j];
        }
    }
    for (std::size_t i = 0; i < Bi.size(); i++) {
        d.Bi[i] += delta.i[i];
        d.Bf[i] += delta.f[i];
        d.Bg[i] += delta.g[i];
        d.Bo[i] += delta.o[i];
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
    Optimize::SGD(W, d.W, learningRate);
    Optimize::SGD(B, d.B, learningRate);

    Optimize::SGD(Wi, d.Wi, learningRate);
    Optimize::SGD(Wg, d.Wg, learningRate);
    Optimize::SGD(Wf, d.Wf, learningRate);
    Optimize::SGD(Wo, d.Wo, learningRate);

    Optimize::SGD(Ui, d.Ui, learningRate);
    Optimize::SGD(Ug, d.Ug, learningRate);
    Optimize::SGD(Uf, d.Uf, learningRate);
    Optimize::SGD(Uo, d.Uo, learningRate);

    Optimize::SGD(Bi, d.Bi, learningRate);
    Optimize::SGD(Bg, d.Bg, learningRate);
    Optimize::SGD(Bf, d.Bf, learningRate);
    Optimize::SGD(Bo, d.Bo, learningRate);
    d.zero();
    return;
}

void RL::LSTM::RMSProp(float learningRate, float rho, float decay)
{
    Optimize::RMSProp(W, s.W, d.W, learningRate, rho, decay);
    Optimize::RMSProp(B, s.B, d.B, learningRate, rho, decay);

    Optimize::RMSProp(Wi, s.Wi, d.Wi, learningRate, rho, decay);
    Optimize::RMSProp(Wg, s.Wg, d.Wg, learningRate, rho, decay);
    Optimize::RMSProp(Wf, s.Wf, d.Wf, learningRate, rho, decay);
    Optimize::RMSProp(Wo, s.Wo, d.Wo, learningRate, rho, decay);

    Optimize::RMSProp(Ui, s.Ui, d.Ui, learningRate, rho, decay);
    Optimize::RMSProp(Ug, s.Ug, d.Ug, learningRate, rho, decay);
    Optimize::RMSProp(Uf, s.Uf, d.Uf, learningRate, rho, decay);
    Optimize::RMSProp(Uo, s.Uo, d.Uo, learningRate, rho, decay);

    Optimize::RMSProp(Bi, s.Bi, d.Bi, learningRate, rho, decay);
    Optimize::RMSProp(Bg, s.Bg, d.Bg, learningRate, rho, decay);
    Optimize::RMSProp(Bf, s.Bf, d.Bf, learningRate, rho, decay);
    Optimize::RMSProp(Bo, s.Bo, d.Bo, learningRate, rho, decay);

    d.zero();
    return;
}

void RL::LSTM::Adam(float learningRate,  float alpha, float beta, float decay)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimize::Adam(W, s.W, v.W, d.W, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(B, s.B, v.B, d.B, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(Wi, s.Wi, v.Wi, d.Wi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Wg, s.Wg, v.Wg, d.Wg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Wf, s.Wf, v.Wf, d.Wf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Wo, s.Wo, v.Wo, d.Wo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(Ui, s.Ui, v.Ui, d.Ui, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Ug, s.Ug, v.Ug, d.Ug, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Uf, s.Uf, v.Uf, d.Uf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Uo, s.Uo, v.Uo, d.Uo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimize::Adam(Bi, s.Bi, v.Bi, d.Bi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Bg, s.Bg, v.Bg, d.Bg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Bf, s.Bf, v.Bf, d.Bf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimize::Adam(Bo, s.Bo, v.Bo, d.Bo, alpha_t, beta_t, learningRate, alpha, beta, decay);
    d.zero();
    return;
}

void RL::LSTM::clamp(float c0, float cn)
{
    Optimize::clamp(W, c0, cn);
    Optimize::clamp(B, c0, cn);

    Optimize::clamp(Wi, c0, cn);
    Optimize::clamp(Wg, c0, cn);
    Optimize::clamp(Wf, c0, cn);
    Optimize::clamp(Wo, c0, cn);

    Optimize::clamp(Ui, c0, cn);
    Optimize::clamp(Ug, c0, cn);
    Optimize::clamp(Uf, c0, cn);
    Optimize::clamp(Uo, c0, cn);

    Optimize::clamp(Bi, c0, cn);
    Optimize::clamp(Bg, c0, cn);
    Optimize::clamp(Bf, c0, cn);
    Optimize::clamp(Bo, c0, cn);
    return;
}

void RL::LSTM::copyTo(LSTM &dst)
{
    for (std::size_t i = 0; i < Wi.rows; i++) {
        for (std::size_t j = 0; j < Wi.cols; j++) {
            dst.Wi(i, j) = Wi(i, j);
            dst.Wg(i, j) = Wg(i, j);
            dst.Wf(i, j) = Wf(i, j);
            dst.Wo(i, j) = Wo(i, j);
        }
    }
    for (std::size_t i = 0; i < Ui.rows; i++) {
        for (std::size_t j = 0; j < Ui.cols; j++) {
            dst.Ui(i, j) = Ui(i, j);
            dst.Ug(i, j) = Ug(i, j);
            dst.Uf(i, j) = Uf(i, j);
            dst.Uo(i, j) = Uo(i, j);
        }
        dst.Bi[i] = Bi[i];
        dst.Bg[i] = Bg[i];
        dst.Bf[i] = Bf[i];
        dst.Bo[i] = Bo[i];
    }
    for (std::size_t i = 0; i < W.rows; i++) {
        for (std::size_t j = 0; j < W.cols; j++) {
            dst.W(i, j) = W(i, j);
        }
        dst.B[i] = B[i];
    }
    return;
}

void RL::LSTM::softUpdateTo(LSTM &dst, float rho)
{
    RL::EMA(dst.Wi, Wi, rho);
    RL::EMA(dst.Wg, Wg, rho);
    RL::EMA(dst.Wf, Wf, rho);
    RL::EMA(dst.Wo, Wo, rho);
    RL::EMA(dst.Ui, Ui, rho);
    RL::EMA(dst.Ug, Ug, rho);
    RL::EMA(dst.Uf, Uf, rho);
    RL::EMA(dst.Uo, Uo, rho);
    RL::EMA(dst.Bi, Bi, rho);
    RL::EMA(dst.Bg, Bg, rho);
    RL::EMA(dst.Bf, Bf, rho);
    RL::EMA(dst.Bo, Bo, rho);
    RL::EMA(dst.W, W, rho);
    RL::EMA(dst.B, B, rho);
    return;
}

void RL::LSTM::test()
{
    LSTM lstm(2, 8, 1, true);
    auto zeta = [](float x, float y) -> float {
        return x*x + y*y + x*y;
    };
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<Mat> data;
    std::vector<Mat> target;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            Mat p(2, 1);
            float x = uniform(Rand::engine);
            float y = uniform(Rand::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Mat q(1, 1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Mat> &batchData,
            std::vector<Mat> &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = selectIndex(Rand::engine);
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Mat> batchData;
        std::vector<Mat> batchTarget;
        sample(batchData, batchTarget, 32);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.Adam(0.001);
    }

    lstm.reset();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            Mat p(2, 1);
            float x = uniform(Rand::engine);
            float y = uniform(Rand::engine);
            float z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            auto s = lstm.forward(p);
            std::cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
