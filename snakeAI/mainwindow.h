#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QPaintEvent>
#include <QTimer>
#include <ctime>
#include <cstdlib>
#include <windows.h>
#include "board.h"
#include "snake.h"
#include "controller.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

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
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
