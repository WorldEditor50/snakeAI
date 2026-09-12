#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QWidget>
#include <QPainter>
#include <QPaintEvent>
#include <QCloseEvent>
#include <QResizeEvent>
#include <QTimer>
#include <QMap>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include "environment.h"
#include "agent.h"


class GameWidget : public QWidget
{
    Q_OBJECT
public:
    explicit GameWidget(QWidget *parent = nullptr);
    void start();
    void stop();
signals:
    void notifyWin(const QString &count);
    void notifyLost(const QString &count);
    void clearReward();
    void sendTotalReward(float r);
    void scale(int value);
    void readyForPaint();
public slots:
    void setBlocks(int value);
    void setAgent(const QString &agentName);
    void setTrainAgent(bool on);
protected:
    void paintEvent(QPaintEvent* ev) override;
    void run();
private:
    QRect getRect(int x, int y);
    void play1();
    void play2();
public:
   Environment env;
private:
    int winCount;
    int lostCount;
    std::atomic<bool> isPlaying;
    std::thread playThread;
    /* Guards every access to `env` and to winCount/lostCount. The play thread
       (run) mutates env.map and env.snake.body while the GUI thread reads them
       in paintEvent and in the slots below; without this the std::deque body
       was read while being modified, which is undefined behaviour. */
    std::mutex envMutex;
};

#endif // GAMEWIDGET_H
