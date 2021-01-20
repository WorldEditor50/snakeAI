#ifndef BOARD_H
#define BOARD_H
#include <vector>
#include <cstdlib>
#include <cmath>
#include "common.h"
using namespace std;
#define BLANK   0
#define BLOCK   1
#define TARGET  2
#define SNAKE   3
class Board
{
public:
    Board(){}
    ~Board(){}
    void init(std::size_t w, std::size_t h);
    void setTarget();
    void setTarget(vector<Point>& body);
    std::size_t width;
    std::size_t height;
    std::size_t rows;
    std::size_t cols;
    std::size_t unitLen;
    std::size_t blockNum;
    int xt;
    int yt;
    vector<vector<int> > map;
};

#endif // BOARD_H
