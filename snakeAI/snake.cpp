#include "snake.h"

void Snake::create(int x, int y)
{
    for (std::size_t i =0; i < 3; i++) {
        body.push_back(Point(x + i, y));
        map(x + i, y) = OBJ_SNAKE;
    }
    return;
}

void Snake::grow(int x, int y)
{
    body.push_front(Point(x, y));
    if (map(x, y) != OBJ_BLOCK) {
        map(x, y) = OBJ_NONE;
    }
    return;
}

void Snake::reset(int rows, int cols)
{
    while (body.size() > 3) {
        Point &p = body.back();
        if (map(p.x, p.y) != OBJ_BLOCK) {
            map(p.x, p.y) = OBJ_NONE;
        }
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
    map(x, y) = OBJ_SNAKE;
    return;
}

void Snake::move(int direct)
{
    Point &p = body.back();
    if (map(p.x, p.y) != OBJ_BLOCK) {
        map(p.x, p.y) = OBJ_NONE;
    }
    body.pop_back();

    int x = body[0].x;
    int y = body[0].y;

    moving(x, y, direct);
    body.push_front(Point(x, y));
    if (map(x, y) != OBJ_BLOCK) {
        map(x, y) = OBJ_SNAKE;
    }
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

