#include "board.h"
#include <iostream>

Board::Board():blockNum(0),agent(map, snake)
{
    xt = -1;
    yt = -1;
    agentMethod.insert(std::pair<std::string, AgentMethod>("astar", &Agent::astarAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("rand", &Agent::randAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("dqn", &Agent::dqnAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("dpg", &Agent::dpgAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("ppo", &Agent::ppoAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("sac", &Agent::sacAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("qlstm", &Agent::qlstmAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("drpg", &Agent::drpgAction));
    act = agentMethod["sac"];
}
void Board::init(size_t w, size_t h)
{

    /* init board parameter */
    this->width = w;
    this->height = h;
    this->unitLen = 5;
    this->blockNum = 0;
    this->rows = width / unitLen - 2;
    this->cols = height / unitLen - 2;
    std::cout<<"rows:"<<rows<<"cols:"<<cols<<std::endl;
    /* init map */
    map = RL::Mat(rows, cols);
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1) {
                map(i, j) = OBJ_BLOCK;
            } else if (j == 0 || j == cols - 1) {
                map(i, j) = OBJ_BLOCK;
            } else {
                map(i, j) = OBJ_NONE;
            }
        }
    }
    /* set block */
    //setBlocks(0);
    /* set target */
    this->setTarget();
    this->snake.create(25, 25);
    return;
}

void Board::setAgent(const std::string &name)
{
    act = agentMethod[name];
    return;
}

void Board::setTrainAgent(bool on)
{
    agent.setTrain(on);
    return;
}

void Board::setTarget()
{
    if (xt >= 0 || yt >= 0) {
        map(xt, yt) = OBJ_NONE;
    }
    int x = rand() % (rows - 1);
    int y = rand() % (cols - 1);
    while (map(x, y) == OBJ_BLOCK) {
        x = rand() % (rows - 1);
        y = rand() % (cols - 1);
    }
    this->xt = x;
    this->yt = y;
    map(xt, yt) = OBJ_TARGET;
    return;
}

void Board::setTarget(const std::deque<Point> &body)
{
    setTarget();
    for (std::size_t i = 0; i < body.size(); i++) {
        if (xt == body[i].x && yt == body[i].y) {
            setTarget();
            i = 0;
        }
    }
    return;
}

void Board::setSnake(const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map(body[i].x, body[i].y) = OBJ_SNAKE;
    }
    return;
}

void Board::clearSnake(const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map(body[i].x, body[i].y) = OBJ_NONE;
    }
    return;
}

void Board::setBlocks(int N)
{
    if (blockNum >= N) {
        return;
    }
    for (int i = 0; i < N - blockNum; i++) {
        int x = rand() % (rows - 10);
        int y = rand() % (cols - 10);
        while (x < 10 || y < 10) {
            x = rand() % rows;
            y = rand() % cols;
        }
        map(x, y) = OBJ_BLOCK;
    }
    blockNum = N;
    return;
}

void Board::moveTo(int direct)
{

}

int Board::play1(float &totalReward)
{
    int x = snake.body[0].x;
    int y = snake.body[0].y;    
    int direct = agent.astarAction(x, y, xt, yt, totalReward);
    snake.move(direct);
    x = snake.body[0].x;
    y = snake.body[0].y;
    /* snake found the target */
    if (x == xt && y == yt) {
        snake.grow(xt, yt);
        setTarget(snake.body);
    }
    /* snake hits block */
    if (map(x, y) == Board::OBJ_BLOCK) {
        snake.reset(rows, cols);
        setTarget(snake.body);
    }
    /* snake hits itself */
#if 0
    for (std::size_t i = 1; i < snake.body.size(); i++) {
        if (x == snake.body[i].x && y == snake.body[i].y) {
            snake.reset(board.rows, board.cols);
            board.setTarget(snake.body);
            break;
        }
    }
#endif
    return 0;
}

int Board::play2(float &totalReward)
{
    int x = snake.body[0].x;
    int y = snake.body[0].y;
    /* make decision */
    int direct = act(agent, x, y, xt, yt, totalReward);
    /* move */
    snake.move(direct);
    x = snake.body[0].x;
    y = snake.body[0].y;
    /* snake found the target */
    if (x == xt && y == yt) {
        if (snake.body.size() < 500) {
           snake.grow(xt, yt);
        }
        setTarget();
        return 1;
    }
    /* snake hits block */
    if (map(x, y) == Board::OBJ_BLOCK) {
        snake.reset(rows, cols);
        setTarget();
        return -1;
    }
    /* snake hits itself */
#if 0
    if (snake.isHitSelf()) {
        snake.reset(board.rows, board.cols);
        board.setTarget();
    }
#endif
    return 0;
}

