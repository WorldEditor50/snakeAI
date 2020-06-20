#include "board.h"

void Board::init()
{
    /* init board parameter */
    this->width = 700;
    this->height = 700;
    this->unitLen = 10;
    this->blockNum = 0;
    this->rows = width / unitLen - 2;
    this->cols = height / unitLen - 2;
    /* init map */
    map.resize(rows + 2);
    for (int i = 0; i < map.size(); i++) {
        map[i].resize(cols + 2);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1) {
                map[i][j] = 1;
            } else if (j == 0 || j == cols - 1) {
                map[i][j] = 1;
            } else {
                map[i][j] = 0;
            }
        }
    }
    /* set block */
    for (int i = 0; i < blockNum; i++) {
        int x = rand() % rows;
        int y = rand() % cols;
        map[x][y] = 1;
    }
    /* set target */
    this->setTarget();
    return;
}

void Board::setTarget()
{
    int x = rand() % (rows - 1);
    int y = rand() % (cols - 1);
    while (map[x][y] == 1) {
        x = rand() % (rows - 1);
        y = rand() % (cols - 1);
    }
    this->xt = x;
    this->yt = y;
    return;
}

void Board::setTarget(vector<Point> &body)
{
    setTarget();
    for (int i = 0; i < body.size(); i++) {
        if (xt == body[i].x && yt == body[i].y) {
            setTarget();
            i = 0;
        }
    }
    return;
}
