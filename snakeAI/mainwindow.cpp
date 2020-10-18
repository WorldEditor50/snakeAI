#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),controller(board.map)
    , ui(new Ui::MainWindow)
{
    QPalette pal;
    pal.setBrush(backgroundRole(), Qt::black);
    setPalette(pal);
    axis = new Axis;
    connect(&controller, &Controller::sigTotalReward, axis, &Axis::addPoint);
    ui->setupUi(this);
    axis->move(1000, 0);
    axis->show();
}

MainWindow::~MainWindow()
{
    controller.dqn.save("./dqn_weights_02");
    controller.dpg.save("./dpg_weights");
    controller.ddpg.save("./ddpg_actor_1", "./ddpg_critic_1");
    controller.bp.save("./bp_weights3");
    axis->close();
    axis->deleteLater();
    delete ui;
}

void MainWindow::init()
{
    int w = 900;
    int h = 900;
    this->setGeometry(100, 50, 1000, 1000);
    this->setFixedSize(w, h);
    this->board.init(w, h);
    this->snake.create(25, 25);
    /* reinforcement learning */
    //controller.dqn.load("./dqn_weights01");
    //controller.dpg.load("./dpg_weights");
    //controller.ddpg.load("./ddpg_actor_1", "./ddpg_critic_1");
    /* supervised learning */
    controller.bp.load("./bp_weights3");
    return;
}

QRect MainWindow::getRect(int x, int y)
{
    int x1 = (x + 1) * board.unitLen;
    int y1 = board.width - (y + 1) * board.unitLen;
    int x2 = (x + 2) * board.unitLen;
    int y2 = board.width - (y + 2) * board.unitLen;
    return QRect(QPoint(x2,y2), QPoint(x1, y1));
}

void MainWindow::paintEvent(QPaintEvent *ev)
{
    QPainter p(this);
    p.setPen(Qt::black);
    p.setBrush(Qt::gray);
    p.setRenderHint(QPainter::Antialiasing);
    /* draw map */
    for (int i=0; i < board.rows; i++) {
        for (int j=0; j < board.cols; j++) {
            if (board.map[i][j] == 1) {
                p.setBrush(Qt::gray);
                QRect rect = getRect(i, j);
                p.drawRect(rect);
            }
        }
    }
    /* draw target */
    p.setBrush(Qt::green);
    p.setPen(Qt::green);
    QRect rect1 = getRect(board.xt, board.yt);
    p.drawRect(rect1);
    /* draw snake */
    p.setBrush(Qt::red);
    p.setPen(Qt::red);
    for (int i = 0; i < snake.body.size(); i++) {
        QRect rect = getRect(snake.body[i].x, snake.body[i].y);
        p.drawRect(rect);
    }
    /* move */
    this->play2();
    Sleep(10);
    return;
}

void MainWindow::play1()
{
    QTimer::singleShot(200, [&]{
        int x = snake.body[0].x;
        int y = snake.body[0].y;
        cout<<"x: "<<x<<" y:"<<y<<endl;
        int direct = controller.AStarAgent(x, y, board.xt, board.yt);
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
        for (int i = 1; i < snake.body.size(); i++) {
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

void MainWindow::play2()
{
    QTimer::singleShot(100, [&]{
        int x = snake.body[0].x;
        int y = snake.body[0].y;
        int direct = 0;
        direct = controller.ppoAgent(x, y, board.xt, board.yt);
        snake.move(direct);
        x = snake.body[0].x;
        y = snake.body[0].y;
        if (x == board.xt && y == board.yt) {
            if (snake.body.size() < 50) {
               snake.add(board.xt, board.yt);
            }
            board.setTarget();
        }
        if (board.map[x][y] == 1) {
            snake.reset(board.rows, board.cols);
            board.setTarget();
        }
        if (snake.check()) {
            //snake.reset(board.rows, board.cols);
            //board.setTarget();
        }
        update();
    });
    return;
}

