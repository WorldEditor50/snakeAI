#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(900, 650);
    QPalette palette;
    palette.setBrush(backgroundRole(), Qt::black);
    palette.setColor(QPalette::WindowText, Qt::white);
    setPalette(palette);
    /* info */
    ui->agentComboBox->addItems(QStringList{"dqn", "sac", "dpg", "ppo", "qlstm",
                                            "drpg", "astar", "rand"});
    /* game */
    connect(ui->agentComboBox, &QComboBox::currentTextChanged,
            ui->gamewidget, &GameWidget::setAgent);
    ui->winValueLabel->setText("0");
    ui->lostValueLabel->setText("0");
    connect(ui->gamewidget, &GameWidget::win,
            ui->winValueLabel, &QLabel::setText, Qt::QueuedConnection);
    connect(ui->gamewidget, &GameWidget::lost,
            ui->lostValueLabel, &QLabel::setText, Qt::QueuedConnection);
    ui->trainCheckBox->setChecked(true);
    connect(ui->trainCheckBox, &QCheckBox::clicked,
            ui->gamewidget, &GameWidget::setTrainAgent);
    /* show reward */
    statisticalWidget = new AxisWidget;
    statisticalWidget->setWindowTitle("Total reward/epoch");
    connect(ui->gamewidget, &GameWidget::sendTotalReward,
            statisticalWidget, &AxisWidget::addPoint, Qt::QueuedConnection);
    statisticalWidget->move(x() + width(), y());
    statisticalWidget->show();
    ui->gamewidget->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if (statisticalWidget != nullptr) {
        ui->gamewidget->stop();
        statisticalWidget->setParent(this);
        statisticalWidget->hide();
    }
    return QMainWindow::closeEvent(ev);
}
