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
    /* Remove every cell the old body occupied. The previous version cleared
       only the segments it popped and then moved body[0] alone, which left
       body[1] and body[2] at their old coordinates and produced a
       disconnected, malformed snake after the first death. */
    for (std::size_t i = 0; i < body.size(); i++) {
        const Point &p = body[i];
        if (map(p.x, p.y) != OBJ_BLOCK) {
            map(p.x, p.y) = OBJ_NONE;
        }
    }
    body.clear();

    /* Pick a head whose 3-segment body (x,y),(x+1,y),(x+2,y) lies on free cells.
       create() places the segments that way, so x must stay <= rows-4. */
    int spanX = static_cast<int>(rows) - 4;
    int spanY = static_cast<int>(cols) - 2;
    if (spanX < 1) {
        spanX = 1;
    }
    if (spanY < 1) {
        spanY = 1;
    }
    int x = 1;
    int y = 1;
    for (int attempt = 0; attempt < 1000; attempt++) {
        int cx = 1 + rand() % spanX;
        int cy = 1 + rand() % spanY;
        bool fits = true;
        for (int i = 0; i < 3; i++) {
            if (map(cx + i, cy) == OBJ_BLOCK) {
                fits = false;
                break;
            }
        }
        if (fits) {
            x = cx;
            y = cy;
            break;
        }
    }
    create(x, y);
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

