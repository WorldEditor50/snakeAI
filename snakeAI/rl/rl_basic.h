#ifndef RL_DEF_H
#define RL_DEF_H
#include <vector>
#include "mat.hpp"

namespace RL {

class Transition
{
public:
    Mat state;
    Mat action;
    Mat nextState;
    float reward;
    bool done;
public:
    Transition(){}
    explicit Transition(const Mat& s,
                        const Mat& a,
                        const Mat& s_,
                        float r,
                        bool d)
        :state(s), action(a), nextState(s_), reward(r), done(d){}
};

class Step
{
public:
    Mat state;
    Mat action;
    float reward;
public:
    Step(){}
    Step(const Mat& s, const Mat& a, float r)
        :state(s), action(a), reward(r) {}
};

}
#endif // RL_DEF_H
