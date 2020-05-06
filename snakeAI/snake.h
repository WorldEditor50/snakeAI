#ifndef SNAKE_H
#define SNAKE_H
#include <iostream>
#include <vector>
#include "common.h"
using namespace std;

class Snake
{
public:
    Snake(){}
    ~Snake(){}
    void create(int x, int y);
    void add(int x, int y);
    void reset(int rows, int cols);
    void move(int direct);
    bool check();
    vector<Point> body;
};

#endif // SNAKE_H
