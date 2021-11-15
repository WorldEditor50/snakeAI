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
    h = Vec(hiddenDim, 0);
    c = Vec(hiddenDim, 0);
    alpha_t = 1;
    beta_t = 1;
    LSTMParam::random();
}

void RL::LSTM::clear()
{
    h.assign(hiddenDim, 0);
    c.assign(hiddenDim, 0);
    return;
}

RL::LSTM::State RL::LSTM::feedForward(const RL::Vec &x, const RL::Vec &_h, const RL::Vec &_c)
{
    /*
                                                        y
                                                        |
                                                       h(t)
                                          c(t)          |
            c(t-1) -->--x-----------------+----------------->--- c(t)
                        |                 |             |
                        |                 |            tanh
                        |                 |             |
                        |          -------x      -------x
                     f  |        i |      | g    | o    |
                        |          |      |      |      |
                     sigmoid    sigmoid  tanh  sigmoid  |
                        |          |      |      |      |
            h(t-1) -->----------------------------      ---->--- h(t)
                        |
                        x(t)
        ft = sigmoid(Wf*xt + Uf*ht-1 + bf);
        it = sigmoid(Wi*xt + Ui*ht-1 + bi);
        gt = tanh(Wg*xt + Ug*ht-1 + bg);
        ot = sigmoid(Wo*xt + Uo*ht-1 + bo);
        ct = ft ⊙ ct-1 + it ⊙ gt
        ht = ot ⊙ tanh(ct)
        yt = sigmoid(W*ht + b)
    */
    State state(hiddenDim, outputDim);
    for (std::size_t i = 0; i < Wi.size(); i++) {
        double sf = 0;
        double si = 0;
        double sg = 0;
        double so = 0;
        for (std::size_t j = 0; j < Wi[0].size(); j++) {
            sf += Wf[i][j] * x[j];
            si += Wi[i][j] * x[j];
            sg += Wg[i][j] * x[j];
            so += Wo[i][j] * x[j];
        }
        for (std::size_t j = 0; j < Ui[0].size(); j++) {
            sf += Uf[i][j] * _h[j];
            si += Ui[i][j] * _h[j];
            sg += Ug[i][j] * _h[j];
            so += Uo[i][j] * _h[j];
        }
        state.f[i] = Sigmoid::_(sf + Bi[i]);
        state.i[i] = Sigmoid::_(si + Bf[i]);
        state.g[i] = Tanh::_(sg + Bg[i]);
        state.o[i] = Sigmoid::_(so + Bo[i]);
        state.c[i] = state.f[i] * _c[i] + state.i[i] * state.g[i];
        state.h[i] = state.o[i] * Tanh::_(state.c[i]);
    }

    for (std::size_t i = 0; i < W.size(); i++) {
        double sy = 0;
        for (std::size_t j = 0; j < W[0].size(); j++) {
            sy += W[i][j] * state.h[j];
        }
        state.y[i] = Linear::_(sy + B[i]);
    }
    return state;
}

void RL::LSTM::forward(const std::vector<RL::Vec> &sequence)
{
    h.assign(hiddenDim, 0);
    c.assign(hiddenDim, 0);
    for (auto &x : sequence) {
        State state = feedForward(x, h, c);
        h = state.h;
        c = state.c;
        states.push_back(state);
    }
    return;
}

RL::Vec RL::LSTM::forward(const RL::Vec &x)
{
    State state = feedForward(x, h, c);
    h = state.h;
    c = state.c;
    return state.y;
}

void RL::LSTM::gradient(const std::vector<RL::Vec> &x,
                        const std::vector<RL::Vec> &yt)
{
    State delta(hiddenDim, outputDim);
    State delta_(hiddenDim, outputDim);
    for (int t = states.size() - 1; t >= 0; t--) {
        /* loss */
        for (std::size_t i = 0; i < delta.y.size(); i++) {
            delta.y[i] = 2 * (states[t].y[i] - yt[t][i]);
        }
        /* backward */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                delta.h[j] += W[i][j] * delta.y[i];
            }
        }
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                delta.h[j] += Ui[i][j] * delta_.i[i];
                delta.h[j] += Uf[i][j] * delta_.f[i];
                delta.h[j] += Ug[i][j] * delta_.g[i];
                delta.h[j] += Uo[i][j] * delta_.o[i];
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
        Vec f_ = t < states.size() - 1 ? states[t + 1].f : Vec(hiddenDim, 0);
        Vec _c = t > 0 ? states[t - 1].c : Vec(hiddenDim, 0);
        for (std::size_t i = 0; i < delta.o.size(); i++) {
            delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::d(states[t].c[i]) +
                    delta_.c[i] * f_[i];
            delta.o[i] = delta.h[i] * Tanh::_(states[t].c[i]) * Sigmoid::d(states[t].o[i]);
            delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::d(states[t].g[i]);
            delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::d(states[t].i[i]);
            delta.f[i] = delta.c[i] * _c[i] * Sigmoid::d(states[t].f[i]);
        }
        /* gradient */
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < W[0].size(); j++) {
                d.W[i][j] += delta.y[i] * Linear::d(states[t].y[i]) * states[t].h[j];
            }
            d.B[i] += delta.y[i] * Linear::d(states[t].y[i]);
        }
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                d.Wi[i][j] += delta.i[i] * x[t][j];
                d.Wf[i][j] += delta.f[i] * x[t][j];
                d.Wg[i][j] += delta.g[i] * x[t][j];
                d.Wo[i][j] += delta.o[i] * x[t][j];
            }
        }
        Vec _h = t > 0 ? states[t - 1].h : Vec(hiddenDim, 0);
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                d.Ui[i][j] += delta.i[i] * _h[j];
                d.Uf[i][j] += delta.f[i] * _h[j];
                d.Ug[i][j] += delta.g[i] * _h[j];
                d.Uo[i][j] += delta.o[i] * _h[j];
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
        delta.zero();
    }
    states.clear();
    return;
}

void RL::LSTM::SGD(double learningRate)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            W[i][j] -= learningRate * d.W[i][j];
        }
        B[i] -= learningRate * d.B[i];
    }
    for (std::size_t i = 0; i < Wi.size(); i++) {
        for (std::size_t j = 0; j < Wi[0].size(); j++) {
            Wi[i][j] -= learningRate * d.Wi[i][j];
            Wf[i][j] -= learningRate * d.Wf[i][j];
            Wg[i][j] -= learningRate * d.Wg[i][j];
            Wo[i][j] -= learningRate * d.Wo[i][j];
        }
    }
    for (std::size_t i = 0; i < Ui.size(); i++) {
        for (std::size_t j = 0; j < Ui[0].size(); j++) {
            Ui[i][j] -= learningRate * d.Ui[i][j];
            Uf[i][j] -= learningRate * d.Uf[i][j];
            Ug[i][j] -= learningRate * d.Ug[i][j];
            Uo[i][j] -= learningRate * d.Uo[i][j];
        }
    }
    for (std::size_t i = 0; i < Bi.size(); i++) {
        Bi[i] -= learningRate * d.Bi[i];
        Bf[i] -= learningRate * d.Bf[i];
        Bg[i] -= learningRate * d.Bg[i];
        Bo[i] -= learningRate * d.Bo[i];
    }
    d.zero();
    return;
}

void RL::LSTM::RMSProp(double learningRate, double rho)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            s.W[i][j] = rho * s.W[i][j] + (1 - rho) * d.W[i][j] * d.W[i][j];
            W[i][j] -= learningRate * d.W[i][j] / (sqrt(s.W[i][j]) + 1e-9);
        }
        s.B[i] = rho * s.B[i] + (1 - rho) * d.B[i] * d.B[i];
        B[i] -= learningRate * d.B[i] / (sqrt(s.B[i]) + 1e-9);
    }
    for (std::size_t i = 0; i < Wi.size(); i++) {
        for (std::size_t j = 0; j < Wi[0].size(); j++) {
            s.Wi[i][j] = rho * s.Wi[i][j] + (1 - rho) * d.Wi[i][j] * d.Wi[i][j];
            s.Wf[i][j] = rho * s.Wf[i][j] + (1 - rho) * d.Wf[i][j] * d.Wf[i][j];
            s.Wg[i][j] = rho * s.Wg[i][j] + (1 - rho) * d.Wg[i][j] * d.Wg[i][j];
            s.Wo[i][j] = rho * s.Wo[i][j] + (1 - rho) * d.Wo[i][j] * d.Wo[i][j];
            Wi[i][j] -= learningRate * d.Wi[i][j] / (sqrt(s.Wi[i][j]) + 1e-9);
            Wf[i][j] -= learningRate * d.Wf[i][j] / (sqrt(s.Wf[i][j]) + 1e-9);
            Wg[i][j] -= learningRate * d.Wg[i][j] / (sqrt(s.Wg[i][j]) + 1e-9);
            Wo[i][j] -= learningRate * d.Wo[i][j] / (sqrt(s.Wo[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Ui.size(); i++) {
        for (std::size_t j = 0; j < Ui[0].size(); j++) {
            s.Ui[i][j] = rho * s.Ui[i][j] + (1 - rho) * d.Ui[i][j] * d.Ui[i][j];
            s.Uf[i][j] = rho * s.Uf[i][j] + (1 - rho) * d.Uf[i][j] * d.Uf[i][j];
            s.Ug[i][j] = rho * s.Ug[i][j] + (1 - rho) * d.Ug[i][j] * d.Ug[i][j];
            s.Uo[i][j] = rho * s.Uo[i][j] + (1 - rho) * d.Uo[i][j] * d.Uo[i][j];
            Ui[i][j] -= learningRate * d.Ui[i][j] / (sqrt(s.Ui[i][j]) + 1e-9);
            Uf[i][j] -= learningRate * d.Uf[i][j] / (sqrt(s.Uf[i][j]) + 1e-9);
            Ug[i][j] -= learningRate * d.Ug[i][j] / (sqrt(s.Ug[i][j]) + 1e-9);
            Uo[i][j] -= learningRate * d.Uo[i][j] / (sqrt(s.Uo[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Bi.size(); i++) {
        s.Bi[i] = rho * s.Bi[i] + (1 - rho) * d.Bi[i] * d.Bi[i];
        s.Bf[i] = rho * s.Bf[i] + (1 - rho) * d.Bf[i] * d.Bf[i];
        s.Bg[i] = rho * s.Bg[i] + (1 - rho) * d.Bg[i] * d.Bg[i];
        s.Bo[i] = rho * s.Bo[i] + (1 - rho) * d.Bo[i] * d.Bo[i];
        Bi[i] -= learningRate * d.Bi[i] / (sqrt(s.Bi[i]) + 1e-9);
        Bf[i] -= learningRate * d.Bf[i] / (sqrt(s.Bf[i]) + 1e-9);
        Bg[i] -= learningRate * d.Bg[i] / (sqrt(s.Bg[i]) + 1e-9);
        Bo[i] -= learningRate * d.Bo[i] / (sqrt(s.Bo[i]) + 1e-9);
    }
    d.zero();
    return;
}

void RL::LSTM::Adam(double learningRate,  double alpha, double beta)
{
    alpha_t *= alpha;
    beta_t *= beta;
    auto AdamMatImpl = [&](Mat &Mw, Mat &Vw, Mat &w, Mat &dW){
        for (std::size_t i = 0; i < w.size(); i++) {
            for (std::size_t j = 0; j < w[0].size(); j++) {
                Mw[i][j] = alpha * Mw[i][j] + (1 - alpha) * dW[i][j];
                Vw[i][j] = beta * Vw[i][j] + (1 - beta) * dW[i][j] * dW[i][j];
                double m = Mw[i][j] / (1 - alpha_t);
                double v = Vw[i][j] / (1 - beta_t);
                w[i][j] -= learningRate * m / (sqrt(v) + 1e-9);
            }
        }
    };
    auto AdamVecImpl = [&](Vec &Mb, Vec &Vb, Vec &b, Vec &dB){
        for (std::size_t i = 0; i < b.size(); i++) {
            Mb[i] = alpha * Mb[i] + (1 - alpha) * dB[i];
            Vb[i] = beta * Vb[i] + (1 - beta) * dB[i] * dB[i];
            double m = Mb[i] / (1 - alpha_t);
            double v = Vb[i] / (1 - beta_t);
            b[i] -= learningRate * m / (sqrt(v) + 1e-9);
        }
    };
    AdamMatImpl(v.W, s.W, W, d.W);
    AdamVecImpl(v.B, s.B, B, d.B);

    AdamMatImpl(v.Wi, s.Wi, Wi, d.Wi);
    AdamMatImpl(v.Wg, s.Wg, Wg, d.Wg);
    AdamMatImpl(v.Wf, s.Wf, Wf, d.Wf);
    AdamMatImpl(v.Wo, s.Wo, Wo, d.Wo);

    AdamMatImpl(v.Ui, s.Ui, Ui, d.Ui);
    AdamMatImpl(v.Ug, s.Ug, Ug, d.Ug);
    AdamMatImpl(v.Uf, s.Uf, Uf, d.Uf);
    AdamMatImpl(v.Uo, s.Uo, Uo, d.Uo);

    AdamVecImpl(v.Bi, s.Bi, Bi, d.Bi);
    AdamVecImpl(v.Bg, s.Bg, Bg, d.Bg);
    AdamVecImpl(v.Bf, s.Bf, Bf, d.Bf);
    AdamVecImpl(v.Bo, s.Bo, Bo, d.Bo);

    d.zero();
    return;
}

void RL::LSTM::test()
{
    srand((unsigned int)time(nullptr));
    LSTM lstm(2, 8, 1, true);
    auto zeta = [](double x, double y) -> double {
        return sin(x*x + y*y);
    };
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<Vec> data;
    std::vector<Vec> target;
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec p(2);
            double x = uniform(Rand::engine);
            double y = uniform(Rand::engine);
            double z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            Vec q(1);
            q[0] = z;
            data.push_back(p);
            target.push_back(q);
        }
    }
    std::uniform_int_distribution<int> selectIndex(0, data.size() - 1);
    auto sample = [&](std::vector<Vec> &batchData,
            std::vector<Vec> &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = selectIndex(Rand::engine);
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Vec> batchData;
        std::vector<Vec> batchTarget;
        sample(batchData, batchTarget, 4);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.RMSProp(0.001);
    }

    lstm.clear();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Vec p(2);
            double x = uniform(Rand::engine);
            double y = uniform(Rand::engine);
            double z = zeta(x, y);
            p[0] = x;
            p[1] = y;
            auto s = lstm.forward(p);
            std::cout<<"x = "<<i<<" y = "<<j<<" z = "<<z<<"  predict: "
                    <<s[0]<<" error:"<<s[0] - z<<std::endl;
        }
    }
    return;
}
