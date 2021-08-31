#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>

namespace RL {

struct Transition
{
    std::vector<double> state;
    std::vector<double> action;
    std::vector<double> nextState;
    double reward;
    bool done;
    Transition(){}
    explicit Transition(std::vector<double>& s, std::vector<double>& a,
               std::vector<double>& s_, double r, bool d)
    {
        state = s;
        action = a;
        nextState = s_;
        reward = r;
        done = d;
    }
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
