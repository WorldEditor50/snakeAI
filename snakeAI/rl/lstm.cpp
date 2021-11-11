#include "lstm.h"
#include <iostream>

RL::Lstm::Lstm(std::size_t inputDim_,
               std::size_t hiddenDim_,
               std::size_t outputDim_,
               bool trainFlag):
    LstmParam(inputDim_, hiddenDim_, outputDim_),
    inputDim(inputDim_), hiddenDim(hiddenDim_), outputDim(outputDim_)
{
    if (trainFlag == true) {
        dP = LstmParam(inputDim_, hiddenDim_, outputDim_);
        Mp = LstmParam(inputDim_, hiddenDim_, outputDim_);
        Vp = LstmParam(inputDim_, hiddenDim_, outputDim_);
    }
    h = Vec(hiddenDim, 0);
    c = Vec(hiddenDim, 0);
    alpha_t = 1;
    beta_t = 1;
    LstmParam::random();
}

void RL::Lstm::clear()
{
    h.assign(hiddenDim, 0);
    c.assign(hiddenDim, 0);
    return;
}

RL::Lstm::State RL::Lstm::feedForward(const RL::Vec &x, const RL::Vec &_h, const RL::Vec &_c)
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

void RL::Lstm::forward(const std::vector<RL::Vec> &sequence)
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

RL::Vec RL::Lstm::forward(const RL::Vec &x)
{
    State state = feedForward(x, h, c);
    h = state.h;
    c = state.c;
    return state.y;
}

void RL::Lstm::gradient(const std::vector<RL::Vec> &x,
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
        for (std::size_t i = 0; i < W[0].size(); i++) {
            for (std::size_t j = 0; j < W.size(); j++) {
                delta.h[i] += W[j][i] * delta.y[j];
            }
        }
        for (std::size_t i = 0; i < Ui[0].size(); i++) {
            for (std::size_t j = 0; j < Ui.size(); j++) {
                delta.h[i] += Ui[j][i] * delta_.i[j];
                delta.h[i] += Uf[j][i] * delta_.f[j];
                delta.h[i] += Ug[j][i] * delta_.g[j];
                delta.h[i] += Uo[j][i] * delta_.o[j];
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
                dP.W[i][j] += delta.y[i] * Linear::d(states[t].y[i]) * states[t].h[j];
            }
            dP.B[i] += delta.y[i] * Linear::d(states[t].y[i]);
        }
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                dP.Wi[i][j] += delta.i[i] * x[t][j];
                dP.Wf[i][j] += delta.f[i] * x[t][j];
                dP.Wg[i][j] += delta.g[i] * x[t][j];
                dP.Wo[i][j] += delta.o[i] * x[t][j];
            }
        }
        Vec _h = t > 0 ? states[t - 1].h : Vec(hiddenDim, 0);
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                dP.Ui[i][j] += delta.i[i] * _h[j];
                dP.Uf[i][j] += delta.f[i] * _h[j];
                dP.Ug[i][j] += delta.g[i] * _h[j];
                dP.Uo[i][j] += delta.o[i] * _h[j];
            }
        }
        for (std::size_t i = 0; i < Bi.size(); i++) {
            dP.Bi[i] += delta.i[i];
            dP.Bf[i] += delta.f[i];
            dP.Bg[i] += delta.g[i];
            dP.Bo[i] += delta.o[i];
        }
        /* next */
        delta_ = delta;
        delta.zero();
    }
    states.clear();
    return;
}

void RL::Lstm::SGD(double learningRate)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            W[i][j] -= learningRate * dP.W[i][j];
        }
        B[i] -= learningRate * dP.B[i];
    }
    for (std::size_t i = 0; i < Wi.size(); i++) {
        for (std::size_t j = 0; j < Wi[0].size(); j++) {
            Wi[i][j] -= learningRate * dP.Wi[i][j];
            Wf[i][j] -= learningRate * dP.Wf[i][j];
            Wg[i][j] -= learningRate * dP.Wg[i][j];
            Wo[i][j] -= learningRate * dP.Wo[i][j];
        }
    }
    for (std::size_t i = 0; i < Ui.size(); i++) {
        for (std::size_t j = 0; j < Ui[0].size(); j++) {
            Ui[i][j] -= learningRate * dP.Ui[i][j];
            Uf[i][j] -= learningRate * dP.Uf[i][j];
            Ug[i][j] -= learningRate * dP.Ug[i][j];
            Uo[i][j] -= learningRate * dP.Uo[i][j];
        }
    }
    for (std::size_t i = 0; i < Bi.size(); i++) {
        Bi[i] -= learningRate * dP.Bi[i];
        Bf[i] -= learningRate * dP.Bf[i];
        Bg[i] -= learningRate * dP.Bg[i];
        Bo[i] -= learningRate * dP.Bo[i];
    }
    dP.zero();
    return;
}

void RL::Lstm::RMSProp(double learningRate, double rho)
{
    for (std::size_t i = 0; i < W.size(); i++) {
        for (std::size_t j = 0; j < W[0].size(); j++) {
            Vp.W[i][j] = rho * Vp.W[i][j] + (1 - rho) * dP.W[i][j] * dP.W[i][j];
            W[i][j] -= learningRate * dP.W[i][j] / (sqrt(Vp.W[i][j]) + 1e-9);
        }
        Vp.B[i] = rho * Vp.B[i] + (1 - rho) * dP.B[i] * dP.B[i];
        B[i] -= learningRate * dP.B[i] / (sqrt(Vp.B[i]) + 1e-9);
    }
    for (std::size_t i = 0; i < Wi.size(); i++) {
        for (std::size_t j = 0; j < Wi[0].size(); j++) {
            Vp.Wi[i][j] = rho * Vp.Wi[i][j] + (1 - rho) * dP.Wi[i][j] * dP.Wi[i][j];
            Vp.Wf[i][j] = rho * Vp.Wf[i][j] + (1 - rho) * dP.Wf[i][j] * dP.Wf[i][j];
            Vp.Wg[i][j] = rho * Vp.Wg[i][j] + (1 - rho) * dP.Wg[i][j] * dP.Wg[i][j];
            Vp.Wo[i][j] = rho * Vp.Wo[i][j] + (1 - rho) * dP.Wo[i][j] * dP.Wo[i][j];
            Wi[i][j] -= learningRate * dP.Wi[i][j] / (sqrt(Vp.Wi[i][j]) + 1e-9);
            Wf[i][j] -= learningRate * dP.Wf[i][j] / (sqrt(Vp.Wf[i][j]) + 1e-9);
            Wg[i][j] -= learningRate * dP.Wg[i][j] / (sqrt(Vp.Wg[i][j]) + 1e-9);
            Wo[i][j] -= learningRate * dP.Wo[i][j] / (sqrt(Vp.Wo[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Ui.size(); i++) {
        for (std::size_t j = 0; j < Ui[0].size(); j++) {
            Vp.Ui[i][j] = rho * Vp.Ui[i][j] + (1 - rho) * dP.Ui[i][j] * dP.Ui[i][j];
            Vp.Uf[i][j] = rho * Vp.Uf[i][j] + (1 - rho) * dP.Uf[i][j] * dP.Uf[i][j];
            Vp.Ug[i][j] = rho * Vp.Ug[i][j] + (1 - rho) * dP.Ug[i][j] * dP.Ug[i][j];
            Vp.Uo[i][j] = rho * Vp.Uo[i][j] + (1 - rho) * dP.Uo[i][j] * dP.Uo[i][j];
            Ui[i][j] -= learningRate * dP.Ui[i][j] / (sqrt(Vp.Ui[i][j]) + 1e-9);
            Uf[i][j] -= learningRate * dP.Uf[i][j] / (sqrt(Vp.Uf[i][j]) + 1e-9);
            Ug[i][j] -= learningRate * dP.Ug[i][j] / (sqrt(Vp.Ug[i][j]) + 1e-9);
            Uo[i][j] -= learningRate * dP.Uo[i][j] / (sqrt(Vp.Uo[i][j]) + 1e-9);
        }
    }
    for (std::size_t i = 0; i < Bi.size(); i++) {
        Vp.Bi[i] = rho * Vp.Bi[i] + (1 - rho) * dP.Bi[i] * dP.Bi[i];
        Vp.Bf[i] = rho * Vp.Bf[i] + (1 - rho) * dP.Bf[i] * dP.Bf[i];
        Vp.Bg[i] = rho * Vp.Bg[i] + (1 - rho) * dP.Bg[i] * dP.Bg[i];
        Vp.Bo[i] = rho * Vp.Bo[i] + (1 - rho) * dP.Bo[i] * dP.Bo[i];
        Bi[i] -= learningRate * dP.Bi[i] / (sqrt(Vp.Bi[i]) + 1e-9);
        Bf[i] -= learningRate * dP.Bf[i] / (sqrt(Vp.Bf[i]) + 1e-9);
        Bg[i] -= learningRate * dP.Bg[i] / (sqrt(Vp.Bg[i]) + 1e-9);
        Bo[i] -= learningRate * dP.Bo[i] / (sqrt(Vp.Bo[i]) + 1e-9);
    }
    dP.zero();
    return;
}

void RL::Lstm::Adam(double learningRate,  double alpha, double beta)
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
    AdamMatImpl(Mp.W, Vp.W, W, dP.W);
    AdamVecImpl(Mp.B, Vp.B, B, dP.B);

    AdamMatImpl(Mp.Wi, Vp.Wi, Wi, dP.Wi);
    AdamMatImpl(Mp.Wg, Vp.Wg, Wg, dP.Wg);
    AdamMatImpl(Mp.Wf, Vp.Wf, Wf, dP.Wf);
    AdamMatImpl(Mp.Wo, Vp.Wo, Wo, dP.Wo);

    AdamMatImpl(Mp.Ui, Vp.Ui, Ui, dP.Ui);
    AdamMatImpl(Mp.Ug, Vp.Ug, Ug, dP.Ug);
    AdamMatImpl(Mp.Uf, Vp.Uf, Uf, dP.Uf);
    AdamMatImpl(Mp.Uo, Vp.Uo, Uo, dP.Uo);

    AdamVecImpl(Mp.Bi, Vp.Bi, Bi, dP.Bi);
    AdamVecImpl(Mp.Bg, Vp.Bg, Bg, dP.Bg);
    AdamVecImpl(Mp.Bf, Vp.Bf, Bf, dP.Bf);
    AdamVecImpl(Mp.Bo, Vp.Bo, Bo, dP.Bo);

    dP.zero();
    return;
}

void RL::Lstm::test()
{
    srand((unsigned int)time(nullptr));
    Lstm lstm(2, 4, 1, true);
    auto zeta = [](double x, double y) -> double {
        return x*x + y*y;
    };
    auto uniform = []()->double{
        int r1 = rand()%10;
        int r2 = rand()%10;
        double s = r1 > r2 ? 1 : -1;
        return s * double(rand()%10000) / 10000;
    };
    std::vector<Vec> data;
    std::vector<Vec> target;
    for (int i = 0; i < 10000; i++) {
        Vec p(2);
        double x = uniform();
        double y = uniform();
        double z = zeta(x, y);
        p[0] = x;
        p[1] = y;
        Vec q(1);
        q[0] = z;
        data.push_back(p);
        target.push_back(q);

    }
    auto sample = [&](std::vector<Vec> &batchData,
            std::vector<Vec> &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % data.size();
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 10000; i++) {
        std::vector<Vec> batchData;
        std::vector<Vec> batchTarget;
        sample(batchData, batchTarget, 512);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.RMSProp(0.0001);
    }

    auto show = [](Vec &y){
        for (std::size_t i = 0; i < y.size(); i++) {
            std::cout<<y[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    };
    lstm.clear();
    for (int i = 0; i < 5; i++) {
        Vec p(2);
        double x = uniform();
        double y = uniform();
        double z = zeta(x, y);
        p[0] = x;
        p[1] = y;
        std::cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<"  predict: ";
        auto s = lstm.forward(p);
        show(s);
    }
    return;
}
