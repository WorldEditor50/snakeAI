#ifndef SNAKE_H
#define SNAKE_H

#include <deque>
#include "common.h"
#include "rl/tensor.hpp"

class Snake
{
public:
    std::deque<Point> body;
    RL::Tensor &map;
public:
    explicit Snake(RL::Tensor &map_)
        :map(map_){}
    explicit Snake(const std::deque<Point> &body_, RL::Tensor &map_)
        :body(body_),map(map_){}
    Snake(const Snake &r):body(r.body),map(r.map){}
    ~Snake(){}
    void create(int x, int y);
    void grow(int x, int y);
    void reset(int rows, int cols);
    void move(int direct);
    bool isHitSelf();

};

#endif // SNAKE_H
