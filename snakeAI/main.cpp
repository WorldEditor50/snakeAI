#include "mainwindow.h"
#include <QApplication>
#include "rl/lstm.h"
#include "rl/gru.h"
#include "rl/mat.hpp"
#include "rl/conv2d.h"
#include "rl/util.hpp"

int main(int argc, char *argv[])
{
#if 1
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
#else
    //BPNN::test();
    //Conv2D::test();
    //matrix::test();
    //Tensor::test();
    return 0;
#endif
}
