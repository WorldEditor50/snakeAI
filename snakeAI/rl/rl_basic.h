#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>
#include "tensor.hpp"

namespace RL {

class Transition
{
public:
    Tensor state;
    Tensor action;
    Tensor nextState;
    float reward;
    bool done;
public:
    Transition(){}
    explicit Transition(const Tensor& s,
                        const Tensor& a,
                        const Tensor& s_,
                        float r,
                        bool d)
        :state(s), action(a), nextState(s_), reward(r), done(d){}
};

class Step
{
public:
    Tensor state;
    Tensor action;
    float reward;
public:
    Step(){}
    Step(const Tensor& s, const Tensor& a, float r)
        :state(s), action(a), reward(r) {}
};

}
#endif // RL_DEF_H
