#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QPaintEvent>
#include <QCloseEvent>
#include <QResizeEvent>
#include <QThread>
#include <QTimer>
#include <QMap>
#include <functional>
#include <ctime>
#include <cstdlib>
#include "board.h"
#include "snake.h"
#include "agent.h"
#include "axis.h"

class GameWidget : public QWidget
{
    Q_OBJECT
public:
    explicit GameWidget(QWidget *parent = nullptr);
signals:
    void win(const QString &count);
    void lost(const QString &count);
    void sendTotalReward(float r);
    void scale(int value);
public slots:
    void setBlocks(int value);
    void setAgent(const QString &agentName);
    void setTrainAgent(bool on);
protected:
    void paintEvent(QPaintEvent* ev) override;
private:
    QRect getRect(int x, int y);
    void play1();
    void play2();
public:
   Board board;
private:
    int winCount;
    int lostCount;
};

#endif // GAMEWIDGET_H
