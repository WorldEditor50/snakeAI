/*
   bench_agent_game — end-to-end benchmark for every agent, driven through the
   real Environment exactly as the GUI does (init(600,600) => 118x118 board,
   setAgent(name), play2 per step).

   Added because "ConvDQN has no noticeable effect in the game" is only
   observable by playing the game: the unit tests never touch the agent
   drivers, and test_convdqn only proves the *network* can learn a synthetic
   task. This measures the thing the user actually sees.

   Reported per phase:
     targets    how many times the snake ate (that is the game score)
     approach%  share of single-cell moves that reduced the distance to the
                target — the direct "is it heading the right way" signal
     meanLen    mean snake length (grows only by eating)
     ms/step    wall clock per game step

   Reference values measured on this machine, 1200 steps:
     astar  99% approach, 2-4 targets/phase      (upper bound)
     rand   ~76% approach, 0-1 targets/phase     (random policy)
     dqn    62 -> 98% approach, learns
     dpg 84->94%, drpg 82->95%, mpg 84->97%, convpg 81->96%
     convdqn  ~50% approach, 0 targets — does NOT learn (see the report)

   Usage: bench_agent_game <agent> <steps> <phases>
*/
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

#include "environment.h"

int main(int argc, char **argv)
{
    const char *agentName = argc > 1 ? argv[1] : "convdqn";
    const int steps  = argc > 2 ? std::atoi(argv[2]) : 1000;
    const int phases = argc > 3 ? std::atoi(argv[3]) : 10;

    std::srand(12345);

    Environment env;
    env.init(600, 600);
    std::cout << "agent=" << agentName
              << " board rows=" << env.rows << " cols=" << env.cols << std::endl;

    env.setAgent(agentName);
    env.setTrainAgent(true);

    const int perPhase = steps / phases;

    auto t0 = std::chrono::steady_clock::now();
    for (int p = 0; p < phases; p++) {
        int eaten = 0;
        int approach = 0, moves = 0;
        double rewardSum = 0;
        long lenSum = 0;
        for (int i = 0; i < perPhase; i++) {
            int before = int(env.snake.body.size());
            int hx = env.snake.body[0].x, hy = env.snake.body[0].y;
            float d1 = 0;
            if (env.xt >= 0) {
                d1 = std::sqrt(float(hx - env.xt)*(hx - env.xt) +
                               float(hy - env.yt)*(hy - env.yt));
            }
            float r = 0;
            env.play2(r);
            int nx = env.snake.body[0].x, ny = env.snake.body[0].y;

            if (int(env.snake.body.size()) > before) {
                eaten++;
            }
            /* Only count single-cell moves; a wall hit teleports the snake back
               to a random spawn, which says nothing about the policy. */
            if (std::abs(nx - hx) + std::abs(ny - hy) == 1 && env.xt >= 0) {
                float d2 = std::sqrt(float(nx - env.xt)*(nx - env.xt) +
                                     float(ny - env.yt)*(ny - env.yt));
                moves++;
                if (d2 < d1) approach++;
            }
            rewardSum += r;
            lenSum += long(env.snake.body.size());
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "phase " << std::setw(2) << p
                  << "  targets=" << std::setw(4) << eaten
                  << "  approach=" << std::fixed << std::setprecision(1)
                  << (moves ? 100.0*approach/moves : 0.0) << "%"
                  << "  meanLen=" << (double(lenSum) / perPhase)
                  << "  meanReward=" << std::setprecision(2)
                  << (rewardSum / perPhase)
                  << "  " << std::setprecision(0)
                  << (secs / ((p + 1) * perPhase) * 1000.0) << " ms/step"
                  << std::endl;
    }
    return 0;
}
