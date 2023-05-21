#include "board.h"
#include <iostream>

void Board::init(size_t w, size_t h)
{
    /* init board parameter */
    this->width = w;
    this->height = h;
    this->unitLen = 5;
    this->blockNum = 0;
    this->rows = width / unitLen - 2;
    this->cols = height / unitLen - 2;
    std::cout<<"rows:"<<rows<<"cols:"<<cols<<std::endl;
    /* init map */
    map = RL::Mat(rows, cols);
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1) {
                map(i, j) = 1;
            } else if (j == 0 || j == cols - 1) {
                map(i, j) = 1;
            } else {
                map(i, j) = 0;
            }
        }
    }
    /* set block */
    //setBlocks(0);
    /* set target */
    this->setTarget();
    return;
}

void Board::setTarget()
{
    int x = rand() % (rows - 1);
    int y = rand() % (cols - 1);
    while (map(x, y) == 1) {
        x = rand() % (rows - 1);
        y = rand() % (cols - 1);
    }
    this->xt = x;
    this->yt = y;
    return;
}

void Board::setTarget(const std::vector<Point> &body)
{
    setTarget();
    for (std::size_t i = 0; i < body.size(); i++) {
        if (xt == body[i].x && yt == body[i].y) {
            setTarget();
            i = 0;
        }
    }
    return;
}

void Board::setBlocks(int N)
{
    if (blockNum >= N) {
        return;
    }
    for (int i = 0; i < N - blockNum; i++) {
        int x = rand() % (rows - 10);
        int y = rand() % (cols - 10);
        while (x < 10 || y < 10) {
            x = rand() % rows;
            y = rand() % cols;
        }
        map(x, y) = 1;
    }
    blockNum = N;
    return;
}
