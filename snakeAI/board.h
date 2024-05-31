#ifndef BOARD_H
#define BOARD_H
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <map>
#include "rl/mat.hpp"
#include "snake.h"
#include "agent.h"
#include "common.h"

#define BLANK   0
#define BLOCK   1
#define TARGET  2
#define SNAKE   3
class Board
{
public:
    enum Object {
        OBJ_NONE = 0,
        OBJ_BLOCK,
        OBJ_TARGET,
        OBJ_SNAKE
    };
    using AgentMethod = std::function<int(Agent&, int, int, int, int, float&)>;
public:
    std::size_t width;
    std::size_t height;
    std::size_t rows;
    std::size_t cols;
    std::size_t unitLen;
    int blockNum;
    int xt;
    int yt;
    RL::Mat map;
    Snake snake;
    Agent agent;
    AgentMethod act;
    std::map<std::string, AgentMethod> agentMethod;
public:
    Board();
    ~Board(){}
    void init(std::size_t w, std::size_t h);
    void setAgent(const std::string &name);
    void setTrainAgent(bool on);
    void setTarget();
    void setTarget(const std::deque<Point> &body);
    void setSnake(const std::deque<Point> &body);
    void clearSnake(const std::deque<Point> &body);
    void setBlocks(int N);
    void moveTo(int direct);
    int play1(float &totalReward);
    int play2(float &totalReward);
};

#endif // BOARD_H
