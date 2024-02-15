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
    board.init(w, h);
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
    int x1 = (x + 1) * board.unitLen;
    int y1 = board.width - (y + 1) * board.unitLen;
    int x2 = (x + 2) * board.unitLen;
    int y2 = board.width - (y + 2) * board.unitLen;
    return QRect(QPoint(x2,y2), QPoint(x1, y1));
}

void GameWidget::paintEvent(QPaintEvent *ev)
{
    Q_UNUSED(ev)
    QPainter painter(this);
    painter.setPen(Qt::black);
    painter.setBrush(Qt::gray);
    painter.setRenderHint(QPainter::Antialiasing);
    /* draw map */
    for (std::size_t i = 0; i < board.rows; i++) {
        for (std::size_t j = 0; j < board.cols; j++) {
            if (board.map(i, j) == Board::OBJ_BLOCK) {
                painter.setBrush(Qt::gray);
                QRect rect = getRect(i, j);
                painter.drawRect(rect);
            }
        }
    }
    /* draw target */
    painter.setBrush(Qt::green);
    painter.setPen(Qt::green);
    QRect rect1 = getRect(board.xt, board.yt);
    painter.drawRect(rect1);
    /* draw snake */
    painter.setBrush(Qt::red);
    painter.setPen(Qt::red);
    for (std::size_t i = 0; i < board.snake.body.size(); i++) {
        QRect rect = getRect(board.snake.body[i].x, board.snake.body[i].y);
        painter.drawRect(rect);
    }
    return QWidget::paintEvent(ev);
}

void GameWidget::run()
{
    /* play */
    while (isPlaying) {
        float r = 0;
        int ret = board.play2(r);
        if (ret > 0) {
            winCount++;
            emit win(QString("%1").arg(winCount));
        } else if (ret < 0) {
            lostCount++;
            emit lost(QString("%1").arg(lostCount));
        }
        emit sendTotalReward(r);
        emit readyForPaint();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return;
}

void GameWidget::setBlocks(int value)
{
    board.setBlocks(value);
    return;
}

void GameWidget::setAgent(const QString &name)
{
    winCount = 0;
    emit win(QString("%1").arg(0));
    lostCount = 0;
    emit lost(QString("%1").arg(0));
    board.setAgent(name.toStdString());
    return;
}

void GameWidget::setTrainAgent(bool on)
{
    board.setTrainAgent(on);
    return;
}

void GameWidget::play1()
{
    QTimer::singleShot(200, [&]{
        float r = 0;
        int ret = board.play1(r);
        if (ret > 0) {
            winCount++;
            emit win(QString("%1").arg(winCount));
        } else if (ret < 0) {
            lostCount++;
            emit lost(QString("%1").arg(lostCount));
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
        int ret = board.play2(r);
        if (ret > 0) {
            winCount++;
            emit win(QString("%1").arg(winCount));
        } else if (ret < 0) {
            lostCount++;
            emit lost(QString("%1").arg(lostCount));
        }
        emit sendTotalReward(r);
        update();
    });
    return;
}

