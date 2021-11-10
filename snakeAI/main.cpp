#include "mainwindow.h"
#include <QApplication>
#include "lstm.h"

int main(int argc, char *argv[])
{
    srand((unsigned int)time(nullptr));
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
