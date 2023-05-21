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
        state.f[i] = Sigmoid::_(state.f[i] + Bf[i]);
        state.i[i] = Sigmoid::_(state.i[i] + Bi[i]);
        state.g[i] =    Tanh::_(state.g[i] + Bg[i]);
        state.o[i] = Sigmoid::_(state.o[i] + Bo[i]);
        state.c[i] = state.f[i] * _c[i] + state.i[i]*state.g[i];
        state.h[i] = state.o[i] * Tanh::_(state.c[i]);
    }
    for (std::size_t i = 0; i < W.rows; i++) {
        for (std::size_t j = 0; j < W.cols; j++) {
            state.y[i] += W(i, j) * state.h[j];
        }
        state.y[i] = Linear::_(state.y[i] + B[i]);
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
        delta.o[i] = delta.h[i] * Tanh::_(states[t].c[i]) * Sigmoid::d(states[t].o[i]);
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
    Optimizer::SGD(W, d.W, learningRate);
    Optimizer::SGD(B, d.B, learningRate);

    Optimizer::SGD(Wi, d.Wi, learningRate);
    Optimizer::SGD(Wg, d.Wg, learningRate);
    Optimizer::SGD(Wf, d.Wf, learningRate);
    Optimizer::SGD(Wo, d.Wo, learningRate);

    Optimizer::SGD(Ui, d.Ui, learningRate);
    Optimizer::SGD(Ug, d.Ug, learningRate);
    Optimizer::SGD(Uf, d.Uf, learningRate);
    Optimizer::SGD(Uo, d.Uo, learningRate);

    Optimizer::SGD(Bi, d.Bi, learningRate);
    Optimizer::SGD(Bg, d.Bg, learningRate);
    Optimizer::SGD(Bf, d.Bf, learningRate);
    Optimizer::SGD(Bo, d.Bo, learningRate);
    d.zero();
    return;
}

void RL::LSTM::RMSProp(float learningRate, float rho, float decay)
{
    Optimizer::RMSProp(W, s.W, d.W, learningRate, rho, decay);
    Optimizer::RMSProp(B, s.B, d.B, learningRate, rho, decay);

    Optimizer::RMSProp(Wi, s.Wi, d.Wi, learningRate, rho, decay);
    Optimizer::RMSProp(Wg, s.Wg, d.Wg, learningRate, rho, decay);
    Optimizer::RMSProp(Wf, s.Wf, d.Wf, learningRate, rho, decay);
    Optimizer::RMSProp(Wo, s.Wo, d.Wo, learningRate, rho, decay);

    Optimizer::RMSProp(Ui, s.Ui, d.Ui, learningRate, rho, decay);
    Optimizer::RMSProp(Ug, s.Ug, d.Ug, learningRate, rho, decay);
    Optimizer::RMSProp(Uf, s.Uf, d.Uf, learningRate, rho, decay);
    Optimizer::RMSProp(Uo, s.Uo, d.Uo, learningRate, rho, decay);

    Optimizer::RMSProp(Bi, s.Bi, d.Bi, learningRate, rho, decay);
    Optimizer::RMSProp(Bg, s.Bg, d.Bg, learningRate, rho, decay);
    Optimizer::RMSProp(Bf, s.Bf, d.Bf, learningRate, rho, decay);
    Optimizer::RMSProp(Bo, s.Bo, d.Bo, learningRate, rho, decay);

    d.zero();
    return;
}

void RL::LSTM::Adam(float learningRate,  float alpha, float beta, float decay)
{
    alpha_t *= alpha;
    beta_t *= beta;
    Optimizer::Adam(W, s.W, v.W, d.W, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(B, s.B, v.B, d.B, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimizer::Adam(Wi, s.Wi, v.Wi, d.Wi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Wg, s.Wg, v.Wg, d.Wg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Wf, s.Wf, v.Wf, d.Wf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Wo, s.Wo, v.Wo, d.Wo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimizer::Adam(Ui, s.Ui, v.Ui, d.Ui, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Ug, s.Ug, v.Ug, d.Ug, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Uf, s.Uf, v.Uf, d.Uf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Uo, s.Uo, v.Uo, d.Uo, alpha_t, beta_t, learningRate, alpha, beta, decay);

    Optimizer::Adam(Bi, s.Bi, v.Bi, d.Bi, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Bg, s.Bg, v.Bg, d.Bg, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Bf, s.Bf, v.Bf, d.Bf, alpha_t, beta_t, learningRate, alpha, beta, decay);
    Optimizer::Adam(Bo, s.Bo, v.Bo, d.Bo, alpha_t, beta_t, learningRate, alpha, beta, decay);
    d.zero();
    return;
}

void RL::LSTM::clamp(float c)
{
    Optimizer::clamp(W, -c, c);
    Optimizer::clamp(B, -c, c);

    Optimizer::clamp(Wi, -c, c);
    Optimizer::clamp(Wg, -c, c);
    Optimizer::clamp(Wf, -c, c);
    Optimizer::clamp(Wo, -c, c);

    Optimizer::clamp(Ui, -c, c);
    Optimizer::clamp(Ug, -c, c);
    Optimizer::clamp(Uf, -c, c);
    Optimizer::clamp(Uo, -c, c);

    Optimizer::clamp(Bi, -c, c);
    Optimizer::clamp(Bg, -c, c);
    Optimizer::clamp(Bf, -c, c);
    Optimizer::clamp(Bo, -c, c);
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
