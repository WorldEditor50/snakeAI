#include "mainwindow.h"
#include <QApplication>
#include "rl/lstm.h"
#include "rl/gru.h"
#include "rl/tensor.hpp"
#include "rl/conv2d.hpp"
#include "rl/util.hpp"

int main(int argc, char *argv[])
{
#if 1
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
#else
    RL::LSTM::test();
    return 0;
#endif
}
