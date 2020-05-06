#include "common.h"

void moving(int& x, int& y, int direct)
{
    switch(direct) {
        case RIGHT:
            x++;
            break;
        case LEFT:
            x--;
            break;
        case UP:
            ++y;
            break;
        case DOWN:
            --y;
            break;
        default:
            break;
    }
    return;
}

void quickSort(int first, int last, int *array)
{
    if (first > last) {
        return;
    }
    int i = first;
    int j = last;
    int ref = array[first];
    while (i < j) {
        while (i < j && array[j] > ref) {
            j--;
        }
        array[i] = array[j];
        while (i < j && array[i] < ref) {
            i++;
        }
        array[j] = array[i];
    }
    array[i] = ref;
    quickSort(first, i - 1, array);
    quickSort(i + 1, last, array);
    return;
}

void bubbleSort(int *array, int len)
{
    for (int i = 0; i < len; i++) {
        for (int j = i + 1; j < len - 1; j++) {
            if (array[i] > array[j]) {
                int tmp = array[i];
                array[i] = array[j];
                array[j] = tmp;
            }
        }
    }
    return;
}
