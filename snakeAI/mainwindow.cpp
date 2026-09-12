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
    ui->agentComboBox->addItems(QStringList{"sac", "dqn", "dpg", "ppo", "trpo", "ddpg",
                                            "drpg", "mpg", "convpg", "convdqn", "bcq", "astar", "rand"});

    /* game */
    connect(ui->agentComboBox, &QComboBox::currentTextChanged,
            ui->gamewidget, &GameWidget::setAgent);
    ui->winValueLabel->setText("0");
    ui->lostValueLabel->setText("0");
    connect(ui->gamewidget, &GameWidget::notifyWin,
            ui->winValueLabel, &QLabel::setText, Qt::QueuedConnection);
    connect(ui->gamewidget, &GameWidget::notifyLost,
            ui->lostValueLabel, &QLabel::setText, Qt::QueuedConnection);
    ui->trainCheckBox->setChecked(true);
    connect(ui->trainCheckBox, &QCheckBox::clicked,
            ui->gamewidget, &GameWidget::setTrainAgent);
    /* The handler used to hard-code setBlocks(100) for BOTH toggle states, so
       unticking the box could not remove the obstacles it had added. */
    connect(ui->blocksCheckBox, &QCheckBox::clicked,
            this, [=](bool checked){
        ui->gamewidget->setBlocks(checked ? 100 : 0);
    });
    /* show reward */
    statisticalWidget = new AxisWidget;
    statisticalWidget->setWindowTitle("Total reward/episode");
    connect(ui->gamewidget, &GameWidget::sendTotalReward,
            statisticalWidget, &AxisWidget::addPoint, Qt::QueuedConnection);
    /* clearReward was emitted on every agent switch but never connected. */
    connect(ui->gamewidget, &GameWidget::clearReward,
            statisticalWidget, &AxisWidget::clearPoints, Qt::QueuedConnection);
    statisticalWidget->move(QPoint(x() + width(), y()));
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
