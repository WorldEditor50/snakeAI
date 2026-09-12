# snakeAI

## 1. Features

Agents selectable at runtime (`Environment::agentMethod`；默认 `sac`）：

| 名称 | 说明 |
|---|---|
| `astar` | A* 贪心策略（不使用学习） |
| `rand` | 基于奖励的随机搜索（MCMC） |
| `dqn` | DQN（MLP，输入为 `observe()` 的 4 维状态） |
| `convdqn` | DQN（卷积，直接吃 118×118 棋盘）**⚠️ 见 §4** |
| `dpg` / `drpg` | 策略梯度 / 策略梯度 + LSTM |
| `mpg` | 策略梯度 + Mamba |
| `convpg` | 卷积策略梯度 |
| `ddpg` | DDPG |
| `ppo` / `trpo` | PPO / TRPO |
| `sac` | SAC |
| `bcq` | BCQ |

> `Agent::supervisedAction`（`bpnn` 监督学习）已实现但**未注册**进 `agentMethod`，
> 因此运行时选不到（见 `docs/optimization_report.md` §11）。

RL 层（`rl/`）是纯 C++、不依赖 Qt；Qt 只出现在 GUI 层。

## 2. Tricks

- 权重用 U(-1, 1) 初始化
- 主要用 RMSProp，ConvDQN 用 Adam
- on-policy 方法使用 layer-norm 与 weight-decay
- **梯度归一化**：`clipGrad = true` 时 `Optimize::*` 做的是
  `dw /= dw.norm2()`（整张量 L2 归一化）。注意这意味着**对损失/advantage 做全局缩放
  会被除掉**，参数更新不变
- 目标网络用 Polyak 软更新

## 3. Reward

- 按位置给奖励（`Environment::reward0`，与代码一致）：

```c++
float Environment::reward0(int xi, int yi, int xn, int yn, int xt, int yt)
{
    /* 撞墙 / 撞到方块 */
    if (map(xn, yn) == OBJ_BLOCK) {
        return -1.5f;
    }
    /* 吃到目标 */
    if (xn == xt && yn == yt) {
        return 1.5f;
    }
    /* 距离塑形：本步缩短了多少距离（正 = 更靠近） */
    float d1 = std::sqrt(float(xi - xt)*(xi - xt) + float(yi - yt)*(yi - yt));
    float d2 = std::sqrt(float(xn - xt)*(xn - xt) + float(yn - yt)*(yn - yt));
    return d1 - d2;
}
```

> **⚠️ 量级不一致**：距离塑形返回的是**原始**距离差，在 118×118 棋盘上可达 **±167**，
> 而撞墙 / 吃到目标只有 **±1.5**。也就是说距离塑形项压倒终止奖励两个数量级。
>
> 策略梯度类算法只通过 advantage 使用奖励，且梯度会被全局归一化，因此不受影响
> （实测 `dpg`/`drpg`/`mpg`/`convpg` 都能正常学会）。
> 但**值方法必须回归奖励**，这是 `convdqn` 问题的候选原因之一（已做截断实验排除，
> 见 `docs/convdqn.md`）。
>
> 若需要"值域受限、与终止奖励同量级"的塑形，用 `0.5f * std::tanh(diff)`
> （`reward4` 就是这么实现的）。修改它**会影响所有算法**，需要重测。

- 每回合累计奖励：

  ![dqn-reward](https://github.com/WorldEditor50/snakeAI/raw/master/reward.png)

## 4. 构建、测试与文档

```powershell
# 配置 + 构建（Clean build ≈ 20 s）
cmake -S snakeAI -B snakeAI\build\verify -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64
cmake --build snakeAI\build\verify

# 测试（13 项，注册进 CTest）
ctest --test-dir snakeAI\build\verify -j 6

# 端到端基准（真实环境，逐阶段学习曲线）
snakeAI\build\verify\bench_agent_game.exe convdqn 2000 10
```

**文档入口**：

| 文档 | 内容 |
|---|---|
| [`snakeAI/task_progress.md`](snakeAI/task_progress.md) | 当前状态、构建/测试/基准命令、下一步建议 |
| [`snakeAI/docs/optimization_report.md`](snakeAI/docs/optimization_report.md) | **问题与优化总记录**（35 项，含实测证据） |
| [`snakeAI/docs/convdqn.md`](snakeAI/docs/convdqn.md) | ConvDQN 专项（含未解决问题） |
| [`snakeAI/docs/temporal_sequence_optimization.md`](snakeAI/docs/temporal_sequence_optimization.md) | 时序信用分配专项 |

**已知未解决**（详见上述文档）：

- `convdqn` 在真实游戏中仍学不会（单元测试 5/5，但 5000 步 approach% 仍 ~50%，
  劣于随机策略的 ~76%）
- `test_convpg` Test 1 / `test_trpo` Test 1 每次都失败，`test_sac` 间歇失败 —— 
  这些在本次修复前就已存在，导致 `ctest` 退出码不能直接当 CI 门禁
