#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <map>
#include "rl/tensor.hpp"
#include "snake.h"
#include "agent.h"
#include "common.h"

#define BLANK   0
#define BLOCK   1
#define TARGET  2
#define SNAKE   3

class Environment
{
public:

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
    RL::Tensor map;
    Snake snake;
    Agent agent;
    AgentMethod act;
    std::map<std::string, AgentMethod> agentMethod;
public:
    Environment();
    ~Environment(){}
    void init(std::size_t w, std::size_t h);
    void setAgent(const std::string &name);
    void setTrainAgent(bool on);
    void setTarget();
    void setTarget(const std::deque<Point> &body);
    void updateMap(const std::deque<Point> &body);
    void updateMap(RL::Tensor &map_, const std::deque<Point> &body);
    void clearSnake(const std::deque<Point> &body);
    void clearSnake(RL::Tensor &map_, const std::deque<Point> &body);
    void setPoint(int x, int y);
    void clearPoint(int x, int y);
    void setBlocks(int N);
    void moveTo(int direct);
    int play1(float &totalReward);
    int play2(float &totalReward);
    float reward0(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward1(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward2(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward3(int xi, int yi, int xn, int yn, int xt, int yt);
    float reward4(int xi, int yi, int xn, int yn, int xt, int yt);
};

#endif // ENVIRONMENT_H
