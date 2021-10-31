#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QPaintEvent>
#include <QCloseEvent>
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
public slots:
    void setBlocks(int value);
    void setAgent(const QString &agentName);
    void setTrain(bool on);
protected:
    void paintEvent(QPaintEvent* ev) override;
private:
    QRect getRect(int x, int y);
    void play1();
    void play2();
public:
    Agent agent;
    using AgentMethod = std::function<int(Agent&, int, int, int, int)>;
private:
    Board board;
    Snake snake;
    QString agentName;
    QMap<QString, AgentMethod> agentMethod;
    int winCount;
    int lostCount;
};

#endif // GAMEWIDGET_H
