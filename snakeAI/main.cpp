#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    srand((unsigned int)time(nullptr));
    QApplication a(argc, argv);
    MainWindow w;
    w.init();
    w.show();
    return a.exec();
}
