#ifndef COMMON_H
#define COMMON_H

#define    RIGHT    0
#define    LEFT     1
#define    UP       2
#define    DOWN     3
enum Object {
    OBJ_NONE = 0,
    OBJ_BLOCK = 1,
    OBJ_SNAKE = 2,
    OBJ_TARGET = 4
};

class Point
{
public:
    Point(){}
    ~Point(){}
    Point(int x_, int y_):x(x_), y(y_){}
    int x;
    int y;
};
void moving(int& x, int& y, int direct);
void quickSort(int first, int last, int* array);
void bubbleSort(int* array, int len);
#endif // COMMON_H
