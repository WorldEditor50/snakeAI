#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPalette palette;
    palette.setBrush(backgroundRole(), Qt::black);
    palette.setColor(QPalette::WindowText, Qt::white);
    setPalette(palette);
    /* info */
    ui->agentComboBox->addItems(QStringList{"dpg", "ppo", "dqn", "qlstm", "drpg", "ddpg", "astar", "rand"});
    /* game */
    connect(ui->agentComboBox, &QComboBox::currentTextChanged,
            ui->gamewidget, &GameWidget::setAgent);
    ui->winValueLabel->setText("0");
    ui->lostValueLabel->setText("0");
    connect(ui->gamewidget, &GameWidget::win, ui->winValueLabel, &QLabel::setText);
    connect(ui->gamewidget, &GameWidget::lost, ui->lostValueLabel, &QLabel::setText);
    ui->trainCheckBox->setChecked(true);
    connect(ui->trainCheckBox, &QCheckBox::clicked, ui->gamewidget, &GameWidget::setTrainAgent);
    /* show reward */
    statisticalWidget = new AxisWidget;
    statisticalWidget->setWindowTitle("Total reward per epoch");
    connect(ui->gamewidget, &GameWidget::sendTotalReward,
            statisticalWidget, &AxisWidget::addPoint);
    statisticalWidget->move(1000, 0);
    statisticalWidget->show();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if (statisticalWidget != nullptr) {
        statisticalWidget->setParent(this);
        statisticalWidget->hide();
    }
    return QMainWindow::closeEvent(ev);
}
