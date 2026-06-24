# snakeAI 程序分析报告

## 一、项目概述

**snakeAI** 是一个使用 **C++** 和 **Qt5/6** 框架开发的贪吃蛇AI训练与演示平台。它从零实现了一个完整的深度学习与强化学习库，并集成了多种主流强化学习算法，用于训练AI自主玩贪吃蛇游戏。

项目分为以下层级：
- **游戏引擎层**（Environment/Snake/GameWidget）— 管理游戏世界、蛇实体、渲染与游戏循环
- **Agent智能体层**（Agent）— 状态观测、11种决策算法注册与调用
- **RL强化学习算法库**（rl/）— 从零实现的DQN/DPG/DDPG/PPO/SAC/QLSTM/DRPG/ConvPG/ConvDQN/BCQ
- **神经网络基础层**（rl/）— Tensor/Layer/Net/Optimizer/Activation/Loss
- **UI可视化层**（MainWindow/AxisWidget）— Qt界面显示游戏状态和训练奖励曲线

---

## 二、项目文件结构

```
snakeAI/
├── main.cpp              # 程序入口
├── mainwindow.*          # 主窗口UI
├── gamewidget.*          # 游戏画布与游戏循环
├── environment.*         # 游戏环境（地图、目标、奖励函数）
├── snake.*               # 蛇实体
├── agent.*               # Agent智能体
├── common.*              # 公共工具
├── axis.*                # 坐标轴控件
├── genetic.*             # 遗传算法（独立工具）
├── CMakeLists.txt        # 构建配置
├── rl/                   # ★ 核心：强化学习与深度学习库
│   ├── rl_basic.h        # 基础数据结构
│   ├── tensor.hpp        # ★ 核心：多维张量类
│   ├── mat.hpp           # 2D矩阵类（早期版本）
│   ├── layer.h           # ★ 神经网络层（FC/LayerNorm/Dropout等）
│   ├── net.hpp           # ★ 神经网络容器
│   ├── ilayer.h          # 层接口
│   ├── activate.h        # 激活函数族
│   ├── loss.h            # 损失函数
│   ├── optimize.h        # 优化器族
│   ├── conv2d.hpp        # 卷积层
│   ├── attention.hpp     # 注意力机制
│   ├── concat.hpp        # 拼接层
│   ├── lstm.*            # LSTM
│   ├── gru.*             # GRU
│   ├── dqn.*             # DQN
│   ├── dpg.*             # DPG
│   ├── ddpg.*            # DDPG
│   ├── ppo.*             # PPO
│   ├── sac.*             # SAC
│   ├── drpg.*            # DRPG
│   ├── qlstm.*           # QLSTM
│   ├── convpg.*          # ConvPG
│   ├── convdqn.*         # ConvDQN
│   ├── bcq.*             # BCQ
│   ├── vae.hpp           # VAE
│   ├── moe.hpp           # MoE
│   ├── parameter.hpp     # 可训练参数封装
│   ├── annealing.hpp     # 退火策略
│   └── util.*            # 工具函数
└── test/test.cpp         # 单元测试
```

---

## 三、各模块详细分析

### 3.1 游戏引擎层

| 模块 | 职责 |
|------|------|
| **Environment** | 管理地图矩阵（rows×cols=118×118，单位5px）、障碍物（blockNum）、目标位置、蛇的引用、Agent方法映射表（map<string, AgentMethod>）、初始化、碰撞检测、5种奖励函数 |
| **Snake** | 用双端队列 `deque<Point>` 存储身体，提供 create/grow/move/reset/isHitSelf 方法 |
| **GameWidget** | Qt画布（600×600px），QPainter 渲染，另起线程运行游戏循环（10ms间隔） |

**关键设计**：
- 地图大小为 `(600/5-2)×(600/5-2) = 118×118` 格
- 边界为障碍物（`OBJ_BLOCK=1`），空地（`OBJ_NONE=0`），目标（`OBJ_TARGET=-1`），蛇身（`OBJ_SNAKE=2`）
- `play2()` 是完整的游戏循环流程
- 奖励函数共5种（reward0~reward4），其中 reward0 使用欧氏距离差，reward2 额外考虑障碍物密度

### 3.2 Agent 智能体层

Agent 聚合了所有 RL 算法实例，通过 `std::map<string, AgentMethod>` 注册函数指针，由下拉菜单选择算法。

**状态表示**（observe 函数）：
- 4维向量：蛇头坐标 (x,y) 和目标坐标 (xt,yt) 归一化到 [-1, 1] 范围
- 卷积网络（ConvPG/ConvDQN）：使用完整地图快照（118×118），reshape 为 (1, 118, 118) 作为输入

**算法实现对应表**：

| 算法 | 训练方式 | 网络结构 | 探索策略 | 经验回放 |
|------|---------|---------|---------|---------|
| **DQN** | Q值回归，软目标网络 | ScaledConcat[16]→Sigmoid | noiseAction(概率噪声) | ✓ |
| **DPG** | REINFORCE策略梯度 | FC→LayerNorm→Softmax | Gumbel-Softmax | × |
| **DDPG** | Actor-Critic | Tanh→LN(Sigmoid)→Softmax + Tanh→TanhNorm(Sigmoid)→Sigmoid | Gumbel-Categorical | ✓ |
| **PPO** | Clip/KL惩罚策略优化 | 同上 | Gumbel-Softmax | ×(trajectory) |
| **SAC** | 最大熵Actor-Critic(4Q) | Actor+4Critics | Gumbel-Softmax | ✓ |
| **QLSTM** | 基于LSTM的DQN | LSTM→TanhNorm(Sigmoid)→Sigmoid | noiseAction | ✓ |
| **DRPG** | 基于LSTM的策略梯度 | LSTM→LN(Sigmoid)→Softmax | Gumbel-Softmax | × |
| **ConvPG** | 基于卷积的策略梯度 | Conv→FC→Softmax | Gumbel-Softmax | × |
| **ConvDQN** | 基于卷积的DQN | Conv→FC→Sigmoid | noiseAction | ✓ |
| **A\* (astar)** | 启发式搜索 | 贪心选择离目标最近的动作 | 无 | × |
| **Rand (rand)** | 模拟退火随机搜索 | 随机采样+退火 | ε-greedy | × |
| **Supervised** | 模仿学习A\* | 4层Sigmoid BPNN | 无 | × |

### 3.3 神经网络基础设施（rl/）

#### 3.3.1 张量系统（tensor.hpp）

核心数据结构 `Tensor_<T>`（通常 `Tensor = Tensor_<float>`）：
- **多维支持**：任意维度张量，通过 `shape/sizes/totalSize` 管理
- **索引系统**：`operator()` 支持变参索引和vector索引，通过预计算步长高效定位
- **视图 & 切片**：`sub()`, `block()`, `embedding()`, `view()`, `flatten()`, `permute()`, `resize()`
- **子张量**：`SubTensor` 内部类
- **矩阵乘法**：内嵌 `MM` 命名空间，支持4种循环序（ikkj/kikj/ikjk/kijk）
- **统计计算**：sum, mean, max, min, argmax, variance, norm2
- **序列化**：toString/fromString 支持CSV格式

#### 3.3.2 神经网络层（ilayer.h + layer.h + net.hpp）

- **iLayer**：定义所有层类型枚举，声明 forward/backward/gradient/SGD/RMSProp/Adam 等纯虚函数
- **Layer<Fn>**：全连接层模板 `o = Fn(W*x + b)`，支持各类优化器
- **Layer特化**：Softmax（Jacobian矩阵）、GELU/Swish/Mish/XTanh（op缓存）
- **Normalization层**：LayerNorm（Pre/Post）、RMSNorm、TanhNorm
- **Dropout**：Bernoulli掩码
- **Net**：网络容器，管理多层串联

#### 3.3.3 激活函数族（activate.h）

Sigmoid(1.702x近似), Tanh, ReLU, LeakyReLU, GELU, Swish, Mish, XTanh, Softplus, Selu, Linear, Softmax

#### 3.3.4 优化器族（optimize.h）

SGD, SGDM, AdaGrad, AdaDelta, RMSProp, Adam（均从零实现）

#### 3.3.5 损失函数（loss.h）

MSE, CrossEntropy, BCE

### 3.4 各强化学习算法实现细节

#### DQN（rl/dqn.*）
- 双网络：QMainNet（训练）+ QTargetNet（软更新）
- 经验回放，Q-Target = r + γ·max(Q_target(s'))
- 探索率指数衰减：1 → 0.99999^n → 0.1

#### DPG（rl/dpg.*）
- 蒙特卡洛轨迹采样，折扣回报 + 基线减方差
- Gumbel-Softmax 可微采样

#### DDPG（rl/ddpg.*）
- Actor-Critic架构（注释："在离散空间可能无效"）
- Gumbel-Categorical 探索
- 软目标网络更新

#### PPO（rl/ppo.*）
- 两种变体：Clip 和 KL惩罚
- Actor + Critic 联合训练

#### SAC（rl/sac.*）
- 最大熵框架：4个Q网络取平均
- 自动温度参数α调整
- KL散度形式的目标函数，带时间偏移

### 3.5 训练流程

1. 用户选择算法 → 切换Agent方法映射
2. 训练循环（play2）：
   - 观测状态（observe）
   - Agent决策动作
   - 模拟移动（simulateMove）得到新状态
   - 计算即时奖励
   - 存储经验/轨迹
   - 执行算法训练
   - 更新游戏画面
3. 游戏结束：碰到障碍物(-1) 或吃到目标(+1)，重置蛇

---

## 四、算法实现中的错误分析

### 🔴 严重错误（运算逻辑错误）

#### 1. kijk 矩阵乘法函数自赋值bug — `rl/tensor.hpp` ~第970行

```cpp
T x1ki = x1ki;  // ← BUG: 未定义行为，自赋值未初始化变量
```

正确应为：
```cpp
T x1ki = x1(k, i);
```

此bug导致 `kijk` 乘法变体产生错误的计算结果。`kijk` 模式在 Net 的反向传播（`kikj` 路径）中被使用，影响所有算法的梯度计算。

#### 2. `cos()` 函数使用 `/=` 而非 `=` — `rl/util.hpp` ~第189行

```cpp
inline Tensor cos(const Tensor& x)
{
    Tensor y(x.shape);
    for (std::size_t i = 0; i < x.totalSize; i++) {
        y[i] /= std::cos(x[i]);  // ← BUG: 应为 = 而非 /=
    }
    return y;
}
```

`y` 初始化为全零，`0 / cos(x) = 0`，此函数永远返回零张量。该函数可在 PositionalEncoder 的正向传播中使用。

#### 3. ScaledConcat/Concat/Attention 更新子层时参数错位 — `rl/concat.hpp`, `rl/attention.hpp`

**RMSProp参数交换** — 子层调用时 `lr` 和 `rho` 被互换：
```cpp
// concat.hpp:147
layers[i].RMSProp(rho, lr, decay, clipGrad);
// 接口签名: RMSProp(lr, rho, decay, clipGrad)
// 实际传入: lr=rho(≈0.9), rho=lr(≈1e-3)
```

**Adam参数完全错位** — 子层调用时参数被轮转：
```cpp
// concat.hpp:166
layers[i].Adam(alpha, beta, alpha_, beta_, lr, decay, clipGrad);
// 接口签名: Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad)
// 实际传入: lr=alpha(≈0.99), α=beta(≈0.9), β=alpha_(≈1.0)
// → 子层学习率被设成≈0.99（本应为≈1e-3），导致训练发散
```

**Attention.hpp 中的同样问题** — `rl/attention.hpp` 第447行和467行：
```cpp
dotProduct[i].RMSProp(rho, lr, decay, clipGrad);
dotProduct[i].Adam(alpha, beta, alpha_, beta_, lr, decay, clipGrad);
```

**影响范围**：使用 ScaledConcat/Attention 作为神经层的 DQN、以及 DDPG/PPO/SAC 中通过 Net 管理的多层网络都会被影响。使用 `ScaledConcat` 作为主网络构建块的 DQN 训练受直接影响。

### 🟠 算法逻辑错误

#### 4. PPO Critic TD目标值计算使用双倍未来奖励 — `rl/ppo.cpp` ~第152-159行

在 PPO 的 `learnWithClipObjective` 和 `learnWithKLpenalty` 中，`trajectory[t].reward` 在第139行已经被覆盖为折扣回报 `G_t = Σγⁱ·r_{t+i}`。但 critic 训练时：

```cpp
// t < end 时:
r[k] = trajectory[t].reward + 0.99*v1[k];
// trajectory[t].reward 已经是 G_t（包含所有未来奖励）
// 加上 γ*V(s') 导致双倍计数未来奖励
// 正确应为:
// r[k] = trajectory[t].reward              (MC目标)
// 或:
// r[k] = immediate_reward_t + 0.99*v1[k]   (TD目标)
```

#### 5. PPO旧策略被持续更新而非冻结 — `rl/ppo.cpp` ~第59-61行

```cpp
if (learningSteps % 16 == 0) {
    actorP.softUpdateTo(actorQ, 0.01);  // actorQ ← 0.99*actorQ + 0.01*actorP
    learningSteps = 0;
}
```

`softUpdateTo(this, dst, alpha)` 的含义是 `dst ← this` 方向更新。因此每次更新使旧策略 `actorQ` 向当前策略 `actorP` 靠近。

在标准PPO中，旧策略（behavior policy，用于计算重要性采样比率 `π_new/π_old`）在一次训练迭代中应保持固定。正确做法应是训练前 `actorQ = actorP`（硬拷贝）并在整个迭代中冻结。

#### 6. SAC温度参数α的梯度计算错误 — `rl/sac.cpp` ~第154-157行

```cpp
const Tensor& prob = x.action;  // 单步动作概率
for (int i = 0; i < actionDim; i++) {
    alpha.g[i] += (RL::entropy(prob[i]) - entropy0)*alpha[i];
}
```

标准SAC的温度参数α更新使用分布熵的梯度：`∇J(α) = -H(π) + H₀`，其中 `H(π) = -Σpᵢ·log(pᵢ)`。

此处：
- 使用单动作概率的熵 `H(pᵢ) = -pᵢ·log(pᵢ)` 而非分布熵
- 乘以自身 `α[i]` 不符合SAC的理论推导
- 应使用完整交叉熵/熵公式计算标量梯度

#### 7. DDPG中Actor损失函数非标准 — `rl/ddpg.cpp` 第80-82行

```cpp
for (std::size_t i = 0; i < actionDim; i++) {
    dLoss[i] = p[i] - p[i]*q[i];  // = p[i]*(1 - q[i])
}
```

标准DDPG策略梯度：最大化 `J = Q(s, π(s))` → 最小化 `L = -Q(s, π(s))`。

此处 `p[i]*(1 - q[i])` 是一个非常规的形式，且已注释说明"在离散动作空间可能无效"。在离散空间应直接最大化所选动作的Q值。

#### 8. SAC中使用的KL散度符号错误 — `rl/util.hpp` ~第242行

```cpp
inline float KL(float p, float q)
{
    return -p*std::log(p/q);
}
```

标准KL散度：`KL(p||q) = p·log(p/q)`。此处返回**负**的KL散度。

在SAC中通过 `loss[i] = p[i] - dL` 的使用方式下（sac.cpp），负号部分抵消，但数学上不正确。应当修正为 `p*log(p/q)` 的正值形式。

#### 9. BCQ 中actor训练使用critic的Q值而不是Q-Target — `rl/bcq.cpp` 第94-104行

BCQ的actor训练中，目标函数是最大化Q值。但代码中使用的是当前critic的Q值而非Q-Target网络的Q值：

```cpp
const Tensor& qi = critics[i].forward(x.state);  // 使用当前critic
```

标准BCQ中应使用clipped Q值（基于目标网络），以避免自举误差。

### 🟡 设计问题

#### 10. DQN/ConvDQN目标网络软更新频率混淆 — `rl/dqn.cpp` 第91-96行

```cpp
if (learningSteps % replaceTargetIter == 0) {
    QMainNet.softUpdateTo(QTargetNet, 0.01);
    learningSteps = 0;  // 清空counting
}
```

清空 `learningSteps` 后，每次 `learn()` 调用 `learningSteps % 0` 导致除零错误... 实际上是 `learningSteps` 被置为0后再 `learningSteps++` 变为1，`1 % replaceTargetIter = 1` 不触发分支。下次 `learningSteps++` 变为2... 直到 `learningSteps == replaceTargetIter` 时触发。所以逻辑上每 `replaceTargetIter` 步更新一次，但`learningSteps` 的初始值从1开始计。

实际上这不算函数错误，因为 `learningSteps` 被重置为0后在下一次 `learn()` 时 `learningSteps++` 变为1，直到 `replaceTargetIter` 才再次更新。这其实是一个设计上的混淆而非bug。

#### 11. MSE损失函数系数 — `rl/loss.h` 第15行

```cpp
loss[i] = 2*d*d;  // 标准MSE: d²，此处为 2d²
```

导数 `∂L/∂y = 4(y-ŷ)`，标准为 `2(y-ŷ)`。梯度被放大了2倍，与优化器学习率耦合，但不影响最终收敛（可被学习率补偿）。

#### 12. SAC `learn()` 中Q-Target计算忽略done标志 — `rl/sac.cpp` ~第117-120行

```cpp
int k = x.action.argmax();
int k_next = QMainNet.forward(x.nextState).argmax();
// ...
qTarget[k] = x.reward + 0.99*v[k_next] - 0.01*std::log(0.25);
```

此处无论 `x.done` 是否为 `true`，都使用 `r + γV(s')` 作为目标。标准实现应在 `done=true` 时使用 `r`（不含未来奖励）。

对比看 `learn()` 中的调用路径，`experienceReplay` 在 `learn()` 的循环中被调用，但 `Transition` 中的 `done` 字段从未被检查。

#### 13. PPO 中 Actor网络更新时使用全梯度而非KL约束 — `rl/ppo.cpp` 第104-106行

```cpp
for (int i = 0; i < actionDim; i++) {
    rTarget[i] = p[i]*prob[i];
}
```

`rTarget` 被设为 `p[i]*prob[i]`（概率×概率），然后通过 CrossEntropy Loss 计算梯度。在PPO的实现中，这个目标应该包含重要性采样比率和优势函数，而非简单的概率乘法。

#### 14. ConvDQN 和 ConvPG 使用克隆地图的状态更新问题 — `agent.cpp` ~第267-293行

ConvPG每次forward后，`state` 指向 `cloneMap`，且在循环中 `state = cloneMap` 是引用赋值。当 `cloneMap` 被 `snake.move()` 修改时，`state` 自动更新。但是 `steps.push_back(Step(state, a, r))` 存储的是对克隆地图的引用，后续梯度计算中 `x[t].state` 的值已被下一帧覆盖。

这在 `reinforce()` 中 `policyNet.forward(x[t].state)` 时使用被覆盖后的地图数据。

---

## 五、架构亮点

1. **从零实现**：完整不依赖第三方深度学习框架，包含Tensor、自动微分、优化器、激活函数、卷积、注意力、LSTM等全套基础设施
2. **丰富算法库**：9种主流RL算法 + A\* + Rand + Supervised，均通过相同Agent接口调用，方便对比
3. **模块化设计**：游戏逻辑与决策逻辑分离，RL库独立可复用
4. **自定义奖励设计**：5种奖励函数从简单欧氏距离到考虑障碍物密度
5. **实时可视化**：Qt界面实时显示游戏状态

## 六、改进建议

1. **修复严重bug**：优先修复 `kijk` 自赋值、`cos()` 操作符错误、子层参数错位三个问题
2. **修正PPO逻辑**：使用正确TD目标、冻结旧策略
3. **修正SAC逻辑**：使用正确的分布熵和KL散度
4. **修正DDPG Actor损失**：使用标准 `-Q(s,π(s))` 形式
5. **修正BCQ逻辑**：使用目标网络Q值训练actor
6. **蛇自碰检测**：当前被注释掉（`#if 0`），蛇不会因自碰撞死亡
7. **状态表示改进**：当前仅4维（蛇头+目标位置），可加入蛇身方向和障碍物信息
8. **线程安全**：游戏循环在独立线程运行，通过Qt信号通信，存在数据竞争风险
9. **代码清理**：未使用变量、调试输出、`genetic.cpp` 与主项目关联不明确

---

## 附录：API参考

### 主要类的public方法

| 类 | 方法 | 说明 |
|----|------|------|
| Tensor | `operator()` | 变参索引访问 |
| Tensor | `shape/sizes/totalSize` | 维度信息 |
| Tensor | `block/embedding/sub` | 切片和子张量 |
| Tensor | `MM::ikkj/kikj/ikjk/kijk` | 矩阵乘法变体 |
| Layer<Fn> | `forward/backward/gradient` | 前向/反向/梯度计算 |
| Layer<Fn> | `SGD/RMSProp/Adam` | 参数更新 |
| Net | `forward(state)` | 前向传播 |
| Net | `backward(e)` | 反向传播 |
| Net | `gradient(x, y)` | 梯度计算 |
| Net | `SGD/RMSProp/Adam()` | 参数更新 |
| Net | `save/load` | 序列化 |
| DQN | `learn(maxMem, replaceTarget, batchSize, lr)` | 训练 |
| PPO | `learnWithClipObjective/KLpenalty` | 两种训练模式 |
| SAC | `learn(maxMem, replaceTarget, batchSize, lr)` | 训练 |
