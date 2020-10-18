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
    void init(int w, int h);
    void setTarget();
    void setTarget(vector<Point>& body);
    int width;
    int height;
    int rows;
    int cols;
    int unitLen;
    int blockNum;
    int xt;
    int yt;
    vector<vector<int> > map;
};

#endif // BOARD_H
