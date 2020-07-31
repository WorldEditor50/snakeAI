#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#if WIN32
#include <windows.h>
#define SLEEP(t) Sleep(t)
#else
#include <unistd.h>
#define SLEEP(t) usleep(t)
#endif
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
    Axis *axis;
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
