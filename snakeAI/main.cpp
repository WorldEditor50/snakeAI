#include "mainwindow.h"
#include <QApplication>
#include "lstm.h"
#include "gru.h"

int main(int argc, char *argv[])
{
#if 1
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
#else
    LSTM::test();
    return 0;
#endif
}
