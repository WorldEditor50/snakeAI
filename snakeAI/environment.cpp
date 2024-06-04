#include "environment.h"
#include <iostream>

Environment::Environment()
    :blockNum(0),snake(map),agent(*this, snake)
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
    agentMethod.insert(std::pair<std::string, AgentMethod>("ddpg", &Agent::ddpgAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("convpg", &Agent::convpgAction));
    agentMethod.insert(std::pair<std::string, AgentMethod>("convdqn", &Agent::convdqnAction));
    act = agentMethod["sac"];
}
void Environment::init(size_t w, size_t h)
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
    map = RL::Tensor(rows, cols);
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
    //setBlocks(60);
    /* set target */
    setTarget();
    snake.create(25, 25);
    return;
}

void Environment::setAgent(const std::string &name)
{
    act = agentMethod[name];
    return;
}

void Environment::setTrainAgent(bool on)
{
    agent.setTrain(on);
    return;
}

void Environment::setTarget()
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
    xt = x;
    yt = y;
    map(xt, yt) = OBJ_TARGET;
    return;
}

void Environment::setTarget(const std::deque<Point> &body)
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

void Environment::updateMap(const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map(body[i].x, body[i].y) = OBJ_SNAKE;
    }
    return;
}

void Environment::updateMap(RL::Tensor &map_, const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map_(body[i].x, body[i].y) = OBJ_SNAKE;
    }
    map_(xt, yt) = OBJ_TARGET;
    return;
}

void Environment::clearSnake(const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map(body[i].x, body[i].y) = OBJ_NONE;
    }
    return;
}

void Environment::clearSnake(RL::Tensor &map_, const std::deque<Point> &body)
{
    for (std::size_t i = 0; i < body.size(); i++) {
        map_(body[i].x, body[i].y) = OBJ_NONE;
    }
    return;
}

void Environment::setPoint(int x, int y)
{
    map(x, y) = OBJ_SNAKE;
}

void Environment::clearPoint(int x, int y)
{
    map(x, y) = OBJ_NONE;
}

void Environment::setBlocks(int N)
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

void Environment::moveTo(int direct)
{

}

int Environment::play1(float &totalReward)
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
    if (map(x, y) == OBJ_BLOCK) {
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

int Environment::play2(float &totalReward)
{
    int x = snake.body[0].x;
    int y = snake.body[0].y;
    /* make decision */
    int k = act(agent, x, y, xt, yt, totalReward);
    /* move */
    snake.move(k);
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
    if (map(x, y) == OBJ_BLOCK) {
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

float Environment::reward0(int xi, int yi, int xn, int yn, int xt, int yt)
{
    /* agent goes out of the map */
    if (map(xn, yn) == 1) {
        return -1;
    }
    /* agent reaches to the target's position */
    if (xn == xt && yn == yt) {
        return 1;
    }
    /* the distance from agent's previous position to the target's position */
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    /* the distance from agent's current position to the target's position */
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return std::sqrt(d1) - std::sqrt(d2);
}

float Environment::reward1(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    float r = std::sqrt(d1) - std::sqrt(d2);
    return r/std::sqrt(1 + r*r);
}

float Environment::reward2(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = (xi - xt) * (xi - xt) + (yi - yt) * (yi - yt);
    float d2 = (xn - xt) * (xn - xt) + (yn - yt) * (yn - yt);
    return std::tanh(d1 - d2);
}

float Environment::reward3(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    float d1 = std::sqrt((xi - xt) * (xi - xt) + (yi - yt) * (yi - yt));
    float d2 = std::sqrt((xn - xt) * (xn - xt) + (yn - yt) * (yn - yt));
    float r = (1 - 2*d2 + d2*d2)/(1 - d2 + d2*d2);
    if (d2 - d1 > 0) {
        r *= -1;
    }
    return r;
}

float Environment::reward4(int xi, int yi, int xn, int yn, int xt, int yt)
{
    if (map(xn, yn) == 1) {
        return -1;
    }
    if (xn == xt && yn == yt) {
        return 1;
    }
    return 0.01;
}
