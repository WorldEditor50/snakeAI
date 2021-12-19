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

}
#endif // RL_DEF_H
