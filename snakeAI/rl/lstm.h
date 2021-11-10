#ifndef LSTM_H
#define LSTM_H

#include "bpnn.h"

namespace RL {

class LstmParam
{
public:
    /* input gate */
    Mat Wi;
    Mat Ui;
    Vec Bi;
    /* generate */
    Mat Wg;
    Mat Ug;
    Vec Bg;
    /* forget gate */
    Mat Wf;
    Mat Uf;
    Vec Bf;
    /* output gate */
    Mat Wo;
    Mat Uo;
    Vec Bo;
    /* predict */
    Mat W;
    Vec B;
public:
    LstmParam(){}
    LstmParam(std::size_t inputDim, std::size_t hiddenDim, std::size_t outputDim)
    {
        Wi = Mat(hiddenDim, Vec(inputDim, 0));
        Wg = Mat(hiddenDim, Vec(inputDim, 0));
        Wf = Mat(hiddenDim, Vec(inputDim, 0));
        Wo = Mat(hiddenDim, Vec(inputDim, 0));

        Ui = Mat(hiddenDim, Vec(hiddenDim, 0));
        Ug = Mat(hiddenDim, Vec(hiddenDim, 0));
        Uf = Mat(hiddenDim, Vec(hiddenDim, 0));
        Uo = Mat(hiddenDim, Vec(hiddenDim, 0));

        Bi = Vec(hiddenDim, 0);
        Bg = Vec(hiddenDim, 0); 
        Bf = Vec(hiddenDim, 0);
        Bo = Vec(hiddenDim, 0);

        W = Mat(outputDim, Vec(hiddenDim, 0));
        B = Vec(outputDim, 0);
    }
    void zero()
    {
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = 0;
                Wg[i][j] = 0;
                Wf[i][j] = 0;
                Wo[i][j] = 0;
            }
        }
        for (std::size_t i = 0; i < Ui.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                Ui[i][j] = 0;
                Ug[i][j] = 0;
                Uf[i][j] = 0;
                Uo[i][j] = 0;
            }
            Bi[i] = 0;
            Bg[i] = 0;
            Bf[i] = 0;
            Bo[i] = 0;
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = 0;
            }
            B[i] = 0;
        }
        return;
    }
    void random()
    {
        auto uniform = []()->double{
            int r1 = rand()%10;
            int r2 = rand()%10;
            double s = r1 > r2 ? 1 : -1;
            return s * double(rand()%10000) / 10000;
        };
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = uniform();
                Wg[i][j] = uniform();
                Wf[i][j] = uniform();
                Wo[i][j] = uniform();
            }
        }
        for (std::size_t i = 0; i < Wi.size(); i++) {
            for (std::size_t j = 0; j < Ui[0].size(); j++) {
                Ui[i][j] = uniform();
                Ug[i][j] = uniform();
                Uf[i][j] = uniform();
                Uo[i][j] = uniform();
            }
            Bi[i] = uniform();
            Bg[i] = uniform();
            Bf[i] = uniform();
            Bo[i] = uniform();
        }
        for (std::size_t i = 0; i < W.size(); i++) {
            for (std::size_t j = 0; j < Wi[0].size(); j++) {
                Wi[i][j] = uniform();
            }
            B[i] = uniform();
        }
        return;
    }
};

class Lstm : public LstmParam
{
public:
    class State
    {
    public:
        Vec i;
        Vec f;
        Vec g;
        Vec o;
        Vec c;
        Vec h;
        Vec y;
    public:
        State(){}
        State(std::size_t hiddenDim, std::size_t outputDim):
            i(Vec(hiddenDim, 0)),f(Vec(hiddenDim, 0)),g(Vec(hiddenDim, 0)),
            o(Vec(hiddenDim, 0)),c(Vec(hiddenDim, 0)),h(Vec(hiddenDim, 0)),
            y(Vec(outputDim, 0)){}
        State(const State &s):
            i(s.i),f(s.f),g(s.g), o(s.o),c(s.o),h(s.h),y(s.y){}
        State& operator = (const State &s)
        {
            if (this == &s) {
                return *this;
            }
            i = s.i;
            f = s.f;
            g = s.g;
            o = s.o;
            c = s.c;
            h = s.h;
            y = s.y;
            return *this;
        }
        void zero()
        {
            for (std::size_t k = 0; k < i.size(); k++) {
                i[k] = 0;
                f[k] = 0;
                g[k] = 0;
                o[k] = 0;
                c[k] = 0;
                h[k] = 0;
            }
            for (std::size_t k = 0; k < y.size(); k++) {
                y[k] = 0;
            }
            return;
        }
        void show()
        {
            std::cout<<"--i--f--g--o--c--h--y--"<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<i[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<f[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<g[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<o[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<c[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < i.size(); k++) {
                std::cout<<h[k]<<" ";
            }
            std::cout<<std::endl;
            for (std::size_t k = 0; k < y.size(); k++) {
                std::cout<<y[k]<<" ";
            }
            std::cout<<std::endl;
            return;
        }
    };
protected:
    std::size_t inputDim;
    std::size_t hiddenDim;
    std::size_t outputDim;
    LstmParam dP;
    LstmParam Mp;
    LstmParam Vp;
    double alpha_t;
    double beta_t;
    /* state */
    std::vector<State> states;
public:
    Lstm(){}
    Lstm(std::size_t inputDim_, std::size_t hiddenDim_, std::size_t outputDim_, bool trainFlag);
    void show();
    State feedForward(const Vec &x);
    void forward(const std::vector<Vec> &seq);
    void gradient(const std::vector<Vec> &seq, const std::vector<Vec> &target);
    void SGD(double learningRate);
    void RMSProp(double learningRate, double rho = 0.9);
    void Adam(double learningRate, double alpha = 0.9, double beta = 0.99);
    static void test();
};

}
#endif // LSTM_H
