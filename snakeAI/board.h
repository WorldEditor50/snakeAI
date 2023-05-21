#ifndef BOARD_H
#define BOARD_H
#include <vector>
#include <cstdlib>
#include <cmath>
#include "mat.hpp"
#include "common.h"

#define BLANK   0
#define BLOCK   1
#define TARGET  2
#define SNAKE   3
class Board
{
public:
    Board():blockNum(0){}
    ~Board(){}
    void init(std::size_t w, std::size_t h);
    void setTarget();
    void setTarget(const std::vector<Point> &body);
    void setBlocks(int N);
public:
    std::size_t width;
    std::size_t height;
    std::size_t rows;
    std::size_t cols;
    std::size_t unitLen;
    int blockNum;
    int xt;
    int yt;
    RL::Mat map;
};

#endif // BOARD_H
