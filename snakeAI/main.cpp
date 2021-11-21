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
    //LSTM::test();
    std::bernoulli_distribution bernoulli;
    for (int i = 0; i < 10; i++) {
        std::cout<<bernoulli(Rand::engine)<<" ";
    }
    std::cout<<std::endl;
    return 0;
#endif
}
