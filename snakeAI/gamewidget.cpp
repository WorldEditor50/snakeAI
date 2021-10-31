#include "gamewidget.h"

GameWidget::GameWidget(QWidget *parent) :
    QWidget(parent),
    agent(this, board.map, snake),
    agentName("astar"),
    winCount(0),
    lostCount(0)
{
    int w = 600;
    int h = 600;
    this->setFixedSize(w, h);
    this->board.init(w, h);
    this->snake.create(25, 25);
    agentMethod.insert("astar", &Agent::astarAction);
    agentMethod.insert("rand", &Agent::randAction);
    agentMethod.insert("dqn", &Agent::dqnAction);
    agentMethod.insert("dpg", &Agent::dpgAction);
    agentMethod.insert("ppo", &Agent::ppoAction);
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
    for (std::size_t i=0; i < board.rows; i++) {
        for (std::size_t j=0; j < board.cols; j++) {
            if (board.map[i][j] == 1) {
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
    for (std::size_t i = 0; i < snake.body.size(); i++) {
        QRect rect = getRect(snake.body[i].x, snake.body[i].y);
        painter.drawRect(rect);
    }
    /* move */
    this->play2();
    QThread::msleep(10);
    return QWidget::paintEvent(ev);
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
    agentName = name;
    return;
}

void GameWidget::setTrain(bool on)
{
    agent.setTrain(on);
}

void GameWidget::play1()
{
    QTimer::singleShot(200, [&]{
        int x = snake.body[0].x;
        int y = snake.body[0].y;
        cout<<"x: "<<x<<" y:"<<y<<endl;
        int direct = agent.astarAction(x, y, board.xt, board.yt);
        snake.move(direct);
        x = snake.body[0].x;
        y = snake.body[0].y;
        if (x == board.xt && y == board.yt) {
            snake.add(board.xt, board.yt);
            board.setTarget(snake.body);
        }
        if (board.map[x][y] == 1) {
            snake.reset(board.rows, board.cols);
            board.setTarget(snake.body);
        }
#if 0
        for (std::size_t i = 1; i < snake.body.size(); i++) {
            if (x == snake.body[i].x && y == snake.body[i].y) {
                snake.reset(board.rows, board.cols);
                board.setTarget(snake.body);
                break;
            }
        }
#endif
        update();
    });
    return;
}

void GameWidget::play2()
{
    QTimer::singleShot(100, [=](){
        int x = snake.body[0].x;
        int y = snake.body[0].y;
        int direct = 0;
        direct = agentMethod[agentName](agent, x, y, board.xt, board.yt);
        snake.move(direct);
        x = snake.body[0].x;
        y = snake.body[0].y;
        if (x == board.xt && y == board.yt) {
            if (snake.body.size() < 500) {
               snake.add(board.xt, board.yt);
            }
            board.setTarget();
            winCount++;
            emit win(QString("%1").arg(winCount));
        }
        if (board.map[x][y] == 1) {
            snake.reset(board.rows, board.cols);
            board.setTarget();
            lostCount++;
            emit lost(QString("%1").arg(lostCount));
        }
#if 0
        if (snake.isHitSelf()) {
            snake.reset(board.rows, board.cols);
            board.setTarget();
        }
#endif
        update();
    });
    return;
}

