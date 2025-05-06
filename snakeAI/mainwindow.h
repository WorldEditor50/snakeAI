#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
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
protected:
    void closeEvent(QCloseEvent *ev) override;
private:
    Ui::MainWindow *ui;
    /* visualize */
    AxisWidget *statisticalWidget;
};
#endif // MAINWINDOW_H
