#include "snake.h"

void Snake::create(int x, int y)
{
    for (std::size_t i =0; i < 3; i++) {
        body.push_back(Point(x + i, y));
    }
    return;
}

void Snake::grow(int x, int y)
{
    body.push_front(Point(x, y));
    return;
}

void Snake::reset(int rows, int cols)
{
    while (body.size() > 3) {
        body.pop_back();
    }
    int x = rand() % (rows - 1);
    int y = rand() % (cols - 1);
    while (x == 0 || y == 0) {
        x = rand() % (rows - 1);
        y = rand() % (cols - 1);
    }
    body[0].x = x;
    body[0].y = y;
    return;
}

void Snake::move(int direct)
{
    body.pop_back();
    int x = body[0].x;
    int y = body[0].y;
    moving(x, y, direct);
    body.push_front(Point(x, y));
    return;
}

bool Snake::isHitSelf()
{
    bool flag = false;
    for (std::size_t i = 1; i < body.size(); i++) {
        if (body[0].x == body[i].x && body[0].y == body[i].y) {
            flag = true;
            break;
        }
    }
    return flag;
}

