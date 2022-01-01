#include "mainwindow.h"
#include <QApplication>
#include "lstm.h"
#include "gru.h"
#include "mat.h"
#include "tensor.h"

int main(int argc, char *argv[])
{
#if 1
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
#else
    //LSTM::test();
    //matrix::test();
    //Tensor::test();
    return 0;
#endif
}
