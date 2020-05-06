#include "snake.h"

void Snake::create(int x, int y)
{
    for (int i =0; i < 3; i++) {
        body.push_back(Point(x + i, y));
    }
    return;
}

void Snake::add(int x, int y)
{
    body.push_back(Point(0, 0));
    for (int i = body.size() - 1; i > 0; i--) {
        body[i].x = body[i - 1].x;
        body[i].y = body[i - 1].y;
    }
    body[0].x = x;
    body[0].y = y;
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
    for (int i = body.size() - 1; i > 0; i--) {
        body[i].x = body[i - 1].x;
        body[i].y = body[i - 1].y;
    }
    moving(body[0].x, body[0].y, direct);
    return;
}

bool Snake::check()
{
    bool flag = false;
    for (int i = 1; i < body.size(); i++) {
        if (body[0].x == body[i].x && body[0].y == body[i].y) {
            flag = true;
            break;
        }
    }
    return flag;
}

