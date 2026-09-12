#include "gamewidget.h"

GameWidget::GameWidget(QWidget *parent) :
    QWidget(parent),
    winCount(0),
    lostCount(0),
    isPlaying(false)
{
    int w = 600;
    int h = 600;
    setFixedSize(w, h);
    connect(this, &GameWidget::readyForPaint, this, [=](){
        update();
    }, Qt::QueuedConnection);

}

void GameWidget::start()
{
    if (isPlaying) {
        return;
    }
    int w = 600;
    int h = 600;
    env.init(w, h);
    isPlaying = true;
    playThread = std::thread(&GameWidget::run, this);
    return;
}

void GameWidget::stop()
{
    if (isPlaying) {
        isPlaying = false;
        playThread.join();
    }
    return;
}

QRect GameWidget::getRect(int x, int y)
{
    int x1 = (x + 1) * env.unitLen;
    int y1 = env.width - (y + 1) * env.unitLen;
    int x2 = (x + 2) * env.unitLen;
    int y2 = env.width - (y + 2) * env.unitLen;
    return QRect(QPoint(x2,y2), QPoint(x1, y1));
}

void GameWidget::paintEvent(QPaintEvent *ev)
{
    /* The play thread mutates env.map / env.snake.body while this runs on the
       GUI thread. Copy everything we need under the lock and paint from the
       snapshot, so the lock is held only for the copy (not for the painting). */
    RL::Tensor mapSnap;
    std::deque<Point> bodySnap;
    int xtSnap = 0;
    int ytSnap = 0;
    std::size_t rowsSnap = 0;
    std::size_t colsSnap = 0;
    int unitLenSnap = 0;
    int widthSnap = 0;
    {
        std::lock_guard<std::mutex> lock(envMutex);
        mapSnap = env.map;
        bodySnap = env.snake.body;
        xtSnap = env.xt;
        ytSnap = env.yt;
        rowsSnap = env.rows;
        colsSnap = env.cols;
        unitLenSnap = env.unitLen;
        widthSnap = static_cast<int>(env.width);
    }
    /* Nothing to draw before Environment::init() has run. */
    if (rowsSnap == 0 || colsSnap == 0 || mapSnap.totalSize == 0) {
        return QWidget::paintEvent(ev);
    }
    auto rectOf = [&](int x, int y) {
        int x1 = (x + 1) * unitLenSnap;
        int y1 = widthSnap - (y + 1) * unitLenSnap;
        int x2 = (x + 2) * unitLenSnap;
        int y2 = widthSnap - (y + 2) * unitLenSnap;
        return QRect(QPoint(x2, y2), QPoint(x1, y1));
    };

    QPainter painter;
    painter.begin(this);
    painter.setPen(Qt::black);
    painter.setBrush(Qt::gray);
    painter.setRenderHint(QPainter::Antialiasing);
    /* draw map */
    for (std::size_t i = 0; i < rowsSnap; i++) {
        for (std::size_t j = 0; j < colsSnap; j++) {
            if (mapSnap(i, j) == OBJ_BLOCK) {
                painter.setBrush(Qt::gray);
                painter.drawRect(rectOf(static_cast<int>(i), static_cast<int>(j)));
            }
        }
    }
    /* draw target */
    painter.setBrush(Qt::green);
    painter.setPen(Qt::green);
    painter.drawRect(rectOf(xtSnap, ytSnap));
    /* draw snake */
    painter.setBrush(Qt::red);
    painter.setPen(Qt::red);
    for (std::size_t i = 0; i < bodySnap.size(); i++) {
        painter.drawRect(rectOf(bodySnap[i].x, bodySnap[i].y));
    }
    painter.end();
    return QWidget::paintEvent(ev);
}

void GameWidget::run()
{
    /* play */
    while (isPlaying) {
        float r = 0;
        int ret = 0;
        int win = 0;
        int lost = 0;
        {
            std::lock_guard<std::mutex> lock(envMutex);
            ret = env.play2(r);
            if (ret > 0) {
                winCount++;
            } else if (ret < 0) {
                lostCount++;
            }
            win = winCount;
            lost = lostCount;
        }
        if (ret > 0) {
            emit notifyWin(QString("%1").arg(win));
        } else if (ret < 0) {
            emit notifyLost(QString("%1").arg(lost));
        }
        emit sendTotalReward(r);
        emit readyForPaint();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return;
}

void GameWidget::setBlocks(int value)
{
    std::lock_guard<std::mutex> lock(envMutex);
    env.setBlocks(value);
    return;
}

void GameWidget::setAgent(const QString &name)
{
    {
        std::lock_guard<std::mutex> lock(envMutex);
        winCount = 0;
        lostCount = 0;
        env.setAgent(name.toStdString());
    }
    emit clearReward();
    emit notifyWin(QString("%1").arg(0));
    emit notifyLost(QString("%1").arg(0));
    return;
}

void GameWidget::setTrainAgent(bool on)
{
    std::lock_guard<std::mutex> lock(envMutex);
    env.setTrainAgent(on);
    return;
}

void GameWidget::play1()
{
    QTimer::singleShot(200, [&]{
        float r = 0;
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(envMutex);
            ret = env.play1(r);
            if (ret > 0) {
                winCount++;
            } else if (ret < 0) {
                lostCount++;
            }
        }
        if (ret > 0) {
            emit notifyWin(QString("%1").arg(winCount));
        } else if (ret < 0) {
            emit notifyLost(QString("%1").arg(lostCount));
        }
        emit sendTotalReward(r);
        update();
    });
    return;
}

void GameWidget::play2()
{
    QTimer::singleShot(100, [=](){
        float r = 0;
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(envMutex);
            ret = env.play2(r);
            if (ret > 0) {
                winCount++;
            } else if (ret < 0) {
                lostCount++;
            }
        }
        if (ret > 0) {
            emit notifyWin(QString("%1").arg(winCount));
        } else if (ret < 0) {
            emit notifyLost(QString("%1").arg(lostCount));
        }
        emit sendTotalReward(r);
        update();
    });
    return;
}
