#ifndef SNAKE_H
#define SNAKE_H
#include <iostream>
#include <vector>
#include <deque>
#include "common.h"

class Snake
{
public:
    std::deque<Point> body;
public:
    Snake(){}
    ~Snake(){}
    void create(int x, int y);
    void grow(int x, int y);
    void reset(int rows, int cols);
    void move(int direct);
    bool isHitSelf();

};

#endif // SNAKE_H
