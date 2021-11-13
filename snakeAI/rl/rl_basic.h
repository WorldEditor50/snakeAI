#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>
#include <random>
#include <ctime>

namespace RL {
using Mat = std::vector<std::vector<double> >;
using Vec = std::vector<double>;

/* activate method */
struct Sigmoid {
    inline static double _(double x) {return exp(x)/(1 + exp(x));}
    inline static double d(double y) {return y*(1 - y);}
};

struct Tanh {
    inline static double _(double x) {return tanh(x);}
    inline static double d(double y) {return 1 - y*y;}
};

struct Relu {
    inline static double _(double x) {return x > 0 ? x : 0;}
    inline static double d(double y) {return y > 0 ? 1 : 0;}
};

struct LeakyRelu {
    inline static double _(double x) {return x > 0 ? x : 0.01*x;}
    inline static double d(double y) {return y > 0 ? 1 : 0.01;}
};

struct Linear {
    inline static double _(double x) {return x;}
    inline static double d(double) {return 1;}
};

/* loss type */
struct Loss
{
    static void MSE(Vec& E, const Vec& O, const Vec& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
             E[i] = 2*(O[i] - y[i]);
        }
        return;
    }
    static void CROSS_EMTROPY(Vec& E, const Vec& O, const Vec& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
            E[i] = -y[i] * log(O[i]);
        }
        return;
    }
    static void BCE(Vec& E, const Vec& O, const Vec& y)
    {
        for (std::size_t i = 0; i < O.size(); i++) {
            E[i] = -(y[i] * log(O[i]) + (1 - y[i]) * log(1 - O[i]));
        }
        return;
    }
};

constexpr static double pi = 3.1415926535898;

struct Transition
{
    std::vector<double> state;
    std::vector<double> action;
    std::vector<double> nextState;
    double reward;
    bool done;
    Transition(){}
    explicit Transition(const std::vector<double>& s,
                        const std::vector<double>& a,
                        const std::vector<double>& s_,
                        double r,
                        bool d)
        :state(s), action(a), nextState(s_), reward(r), done(d){}
};

struct Step
{
    std::vector<double> state;
    std::vector<double> action;
    double reward;
    Step(){}
    Step(std::vector<double>& s, std::vector<double>& a, double r)
        :state(s), action(a), reward(r) {}
};
class Rand
{
public:
    static std::default_random_engine engine;
};

double uniformDistribution(double sup, double inf);
int uniformDistribution(int sup, int inf);
double normalDistribution(double mu, double sigma, double sup, double inf);
void normalDistribution(double mu, double sigma, double sup, double inf, std::vector<double> &x, std::size_t N);
double clip(double x, double sup, double inf);
int argmax(const std::vector<double> &x);
int argmin(const std::vector<double> &x);
double max(const std::vector<double> &x);
double min(const std::vector<double> &x);
double sum(const std::vector<double> &x);
double mean(const std::vector<double> &x);
double variance(const std::vector<double> &x);
double covariance(const std::vector<double>& x1, const std::vector<double>& x2);
void zscore(std::vector<double> &x);
void normalize(std::vector<double> &x);
double dotProduct(const std::vector<double>& x1, const std::vector<double>& x2);
}
#endif // RL_DEF_H
