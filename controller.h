#ifndef CONTROLLER_H
#define CONTROLLER_H
#include <vector>
#include "dqn.h"
#include "policyGradient.h"
#include "common.h"
using namespace std;
using namespace ML;
class Controller
{
public:
    Controller(vector<vector<int> >& map);
    void setState(vector<double>& statex,int x, int y, int xt, int yt);
    int simpleAgent(int x, int y, int xt, int yt);
    int simpleAgent2(vector<Point>& body, int xt, int yt);
    int randomSearchAgent(int x, int y, int xt, int yt);
    int dqnAgent(int x, int y, int xt, int yt);
    int dpgAgent(int x, int y, int xt, int yt);
    int bpAgent(int x, int y, int xt, int yt);
    double reward(int xi, int yi, int xn, int yn, int xt, int yt);
    bool move(int& x, int& y, int direct);
    int maxAction(vector<double>& action);
    int MonteCarloAgent(vector<Point>& body, int xt, int yt);
    double reward(int xi, int yi, int xn, int yn, int xt, int yt, vector<Point>& body);
    bool check(vector<Point>& body);
    void trymove(vector<Point>& body, int direct);
    vector<vector<int> >& map;
    vector<double> state;
    vector<double> nextState;
    DQNet dqn;
    BPNet bp;
    DPGNet dpg;
};

#endif // CONTROLLER_H
