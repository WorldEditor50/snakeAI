#include "lstm.h"

RL::LSTM::LSTM(std::size_t inputDim_,
               std::size_t hiddenDim_,
               std::size_t outputDim_,
               bool trainFlag)
    :inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    type = iLayer::LAYER_LSTM;
    wi = Tensor(hiddenDim, inputDim);
    wg = Tensor(hiddenDim, inputDim);
    wf = Tensor(hiddenDim, inputDim);
    wo = Tensor(hiddenDim, inputDim);

    ui = Tensor(hiddenDim, hiddenDim);
    ug = Tensor(hiddenDim, hiddenDim);
    uf = Tensor(hiddenDim, hiddenDim);
    uo = Tensor(hiddenDim, hiddenDim);

    bi = Tensor(hiddenDim, 1);
    bg = Tensor(hiddenDim, 1);
    bf = Tensor(hiddenDim, 1);
    bo = Tensor(hiddenDim, 1);

    w = Tensor(outputDim, hiddenDim);
    b = Tensor(outputDim, 1);

    std::vector<Tensor*> weights = {&wi, &wg, &wf, &wo,
                                 &ui, &ug, &uf, &uo,
                                 &bi, &bg, &bf, &bo,
                                 &w, &b};
    for (std::size_t i = 0; i < weights.size(); i++) {
        RL::uniformRand(*weights[i], -1, 1);
    }
    if (trainFlag == true) {
        g = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        v = LSTMParam(inputDim_, hiddenDim_, outputDim_);
        s = LSTMParam(inputDim_, hiddenDim_, outputDim_);
    }
    h = Tensor(hiddenDim, 1);
    c = Tensor(hiddenDim, 1);
    o = Tensor(outputDim, 1);
}

void RL::LSTM::reset()
{
    h.zero();
    c.zero();
    cacheX.clear();
    cacheE.clear();
    states.clear();
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

RL::Tensor &RL::LSTM::forward(const RL::Tensor &x, bool inference)
{
    State state = feedForward(x, h, c);
    h = state.h;
    c = state.c;
    o = state.y;
    if (inference == false) {
        cacheX.push_back(x);
        states.push_back(state);
    }
    return o;
}

void RL::LSTM::backwardAtTime(int t,
                         const RL::Tensor &x,
                         const RL::Tensor &E,
                         State &delta_)
{
    State delta(hiddenDim, outputDim);
    Tensor::MM::kijk(delta.h, w, E);

    Tensor::MM::kijk(delta.h, ui, delta_.i);
    Tensor::MM::kijk(delta.h, uf, delta_.f);
    Tensor::MM::kijk(delta.h, ug, delta_.g);
    Tensor::MM::kijk(delta.h, uo, delta_.o);

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
        delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::df(states[t].c[i]) + delta_.c[i] * f_[i];
        delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::df(states[t].o[i]);
        delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::df(states[t].g[i]);
        delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::df(states[t].i[i]);
        delta.f[i] = delta.c[i] * _c[i] * Sigmoid::df(states[t].f[i]);
    }
    /* gradient */
    for (std::size_t i = 0; i < w.shape[0]; i++) {
        for (std::size_t j = 0; j < w.shape[1]; j++) {
            g.w(i, j) += E[i] * states[t].y[i] * states[t].h[j];
        }
        g.b[i] += E[i] * states[t].y[i];
    }

    Tensor::MM::ikjk(g.wi, delta.i, x);
    Tensor::MM::ikjk(g.wf, delta.f, x);
    Tensor::MM::ikjk(g.wg, delta.g, x);
    Tensor::MM::ikjk(g.wo, delta.o, x);
    Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
    Tensor::MM::ikjk(g.ui, delta.i, _h);
    Tensor::MM::ikjk(g.uf, delta.f, _h);
    Tensor::MM::ikjk(g.ug, delta.g, _h);
    Tensor::MM::ikjk(g.uo, delta.o, _h);

    g.bi += delta.i;
    g.bf += delta.f;
    g.bg += delta.g;
    g.bo += delta.o;
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

void RL::LSTM::cacheError(const RL::Tensor &e)
{
    cacheE.push_back(e);
    return;
}

void RL::LSTM::SGD(float lr)
{   
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    Optimize::SGD(w, g.w, lr);
    Optimize::SGD(b, g.b, lr);

    Optimize::SGD(wi, g.wi, lr);
    Optimize::SGD(wg, g.wg, lr);
    Optimize::SGD(wf, g.wf, lr);
    Optimize::SGD(wo, g.wo, lr);

    Optimize::SGD(ui, g.ui, lr);
    Optimize::SGD(ug, g.ug, lr);
    Optimize::SGD(uf, g.uf, lr);
    Optimize::SGD(uo, g.uo, lr);

    Optimize::SGD(bi, g.bi, lr);
    Optimize::SGD(bg, g.bg, lr);
    Optimize::SGD(bf, g.bf, lr);
    Optimize::SGD(bo, g.bo, lr);
    g.zero();
    return;
}

void RL::LSTM::RMSProp(float lr, float rho, float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();
    Optimize::RMSProp(w, s.w, g.w, lr, rho, decay);
    Optimize::RMSProp(b, s.b, g.b, lr, rho, decay);

    Optimize::RMSProp(wi, s.wi, g.wi, lr, rho, decay);
    Optimize::RMSProp(wg, s.wg, g.wg, lr, rho, decay);
    Optimize::RMSProp(wf, s.wf, g.wf, lr, rho, decay);
    Optimize::RMSProp(wo, s.wo, g.wo, lr, rho, decay);

    Optimize::RMSProp(ui, s.ui, g.ui, lr, rho, decay);
    Optimize::RMSProp(ug, s.ug, g.ug, lr, rho, decay);
    Optimize::RMSProp(uf, s.uf, g.uf, lr, rho, decay);
    Optimize::RMSProp(uo, s.uo, g.uo, lr, rho, decay);

    Optimize::RMSProp(bi, s.bi, g.bi, lr, rho, decay);
    Optimize::RMSProp(bg, s.bg, g.bg, lr, rho, decay);
    Optimize::RMSProp(bf, s.bf, g.bf, lr, rho, decay);
    Optimize::RMSProp(bo, s.bo, g.bo, lr, rho, decay);

    g.zero();
    return;
}

void RL::LSTM::Adam(float lr, float alpha, float beta,
                    float alpha_, float beta_,
                    float decay, bool clipGrad)
{
    backward(cacheX, cacheE);
    cacheX.clear();
    cacheE.clear();

    Optimize::Adam(w, s.w, v.w, g.w, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(b, s.b, v.b, g.b, alpha_, beta_, lr, alpha, beta, decay);

    Optimize::Adam(wi, s.wi, v.wi, g.wi, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(wg, s.wg, v.wg, g.wg, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(wf, s.wf, v.wf, g.wf, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(wo, s.wo, v.wo, g.wo, alpha_, beta_, lr, alpha, beta, decay);

    Optimize::Adam(ui, s.ui, v.ui, g.ui, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(ug, s.ug, v.ug, g.ug, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(uf, s.uf, v.uf, g.uf, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(uo, s.uo, v.uo, g.uo, alpha_, beta_, lr, alpha, beta, decay);

    Optimize::Adam(bi, s.bi, v.bi, g.bi, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(bg, s.bg, v.bg, g.bg, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(bf, s.bf, v.bf, g.bf, alpha_, beta_, lr, alpha, beta, decay);
    Optimize::Adam(bo, s.bo, v.bo, g.bo, alpha_, beta_, lr, alpha, beta, decay);
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

void RL::LSTM::copyTo(iLayer *layer)
{
    LSTM &dst = *static_cast<LSTM*>(layer);
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

void RL::LSTM::softUpdateTo(iLayer *layer, float rho)
{
    LSTM &dst = *static_cast<LSTM*>(layer);
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
    for (int i = 0; i < 1000; i++) {
        lstm.reset();
        for (int j = 0; j < 16; j++) {
            int k = selectIndex(Random::engine);
            Tensor& out = lstm.forward(data[k]);
            lstm.cacheError(Loss::MSE(out, target[k]));
        }
        lstm.RMSProp(0.9, 1e-3, 0, true);
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
