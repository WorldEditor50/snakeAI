#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QPaintEvent>
#include <QTimer>
#include <ctime>
#include <cstdlib>
#include "board.h"
#include "snake.h"
#include "controller.h"
#include "axis.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void init();
    QRect getRect(int x, int y);
    void paintEvent(QPaintEvent* ev);
    void play1();
    void play2();
    Board board;
    Snake snake;
    Controller controller;
    /* visualize */
    Axis *axis;
};
#endif // MAINWINDOW_H
