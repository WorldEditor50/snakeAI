#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>
#include <random>

namespace RL {
using Mat = std::vector<std::vector<double> >;
using Vec = std::vector<double>;
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
double uniformDistribution(double sup, double inf);
double normalDistribution(double mu, double sigma, double bound);
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
