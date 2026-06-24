# QLSTM (LSTM-DQN) 算法优化报告

## 概述

对 `rl/qlstm.h` 和 `rl/qlstm.cpp` 进行了系统性优化，修复了原始实现中与DQN标准架构及LSTM序列学习不符的问题。

---

## 原始实现的问题分析

### 🔴 1. 训练时：随机采样破坏LSTM时序依赖（最严重）

```cpp
// 原始实现（严重错误）：
lstm->reset();
for (std::size_t i = 0; i < batchSize; i++) {
    int k = uniform(Random::engine);
    experienceReplay(memories[k]);  // ❌ 随机采样，时序断裂！
}
```

每次 `forward` 前调用 `lstm->reset()`，然后随机从回放缓冲区抽取独立的transition。LSTM的隐藏状态 `h` 和 `c` 被重置到0，随机样本之间**没有任何时序关联**。

**后果**：LSTM退化为普通MLP，丧失了序列建模能力。

**修正**：使用**序列经验回放**（Sequence-based Experience Replay），从完整episode中抽取连续子序列进行BPTT。

### 🟠 2. 目标网络更新策略有缺陷

```cpp
// 原始实现：
if (learningSteps % replaceTargetIter == 0) {
    QMainNet.softUpdateTo(QTargetNet, 0.01);
    learningSteps = 0;
}
```

每256步才软更新一次，且更新后 `learningSteps` 置零。但软更新（Polyak=0.01）应该**每步都进行**。硬更新才需要间隔更新。混合策略导致目标网络更新不足。

**修正**：每轮学习都使用 `softUpdateTo(tau=5e-3)`。

### 🟠 3. LSTM状态管理混乱

```cpp
// 原始实现：
h = lstm->h;  // 保存最后一步的h
c = lstm->c;  // 保存最后一步的c
// ... 训练后LSTM状态被reset并随机采样破坏
```

每次 `learn()` 开始时保存LSTM状态到 `h/c`，但 `action()` 中又写回 `lstm->h = h; lstm->c = c`。这些状态在推理时可能起作用，但在训练时被随机采样全部覆盖，实际上无意义。

**修正**：保留 `h/c` 用于推理时的状态传播，训练时每个序列独立reset。

### 🟡 4. 缺乏Double DQN

```cpp
// 原始实现：
int k = QMainNet.forward(x.nextState, true).argmax();
Tensor &v = QTargetNet.forward(x.nextState, true);
qTarget[i] = x.reward + gamma * v[k];
```

正确实现了Double DQN（用QMainNet选动作，QTargetNet估值），这个部分是好的。但训练流程破坏了其效果。

### 🟡 5. 优化器学习率未调优

使用 `RMSProp(learningRate, 0.9, 0)`，与DQN实现一致。对于LSTM训练，梯度更不稳定，采用 `learningRate * 0.5` 更安全。

---

## 优化内容

### 1. 序列经验回放（核心优化）

**数据结构**：
- 新增 `seqEnds`（`deque<bool>`）跟踪每个transition的episode结束标记
- 新增 `currentSeqId` 记录当前episode编号

**采样策略**：
1. 扫描 `seqEnds` 找出所有完整episode的起始位置和长度
2. 随机选择一个episode
3. 从该episode中随机抽取长度为 `seqLen=8` 的连续子序列
4. LSTM状态重置 → 顺次forward子序列 → 每步累积梯度 → BPTT反向传播

```
Episode: [s₀→s₁→s₂→...→s₁₅→done]
                  ↓
子序列:  [s₃→s₄→s₅→s₆→s₇→s₈→s₉→s₁₀]  (seqLen=8)
                    ↓
LSTM前向: h₀→h₁→...→h₇, 每步计算Q-target并累积梯度
                    ↓
BPTT: 梯度沿时间步反向传播
```

### 2. 目标网络分离

```cpp
// 新增独立目标LSTM网络
lstmTarget = LSTM::_(stateDim_, hiddenDim_, hiddenDim_, false);
QTargetNet = Net(lstmTarget,
                 TanhNorm<Sigmoid>::_(...),
                 Layer<Sigmoid>::_(...));
```

原始实现中 `QTargetNet` 和 `QMainNet` 共用同一个 `lstm` 指针，导致目标网络forward会修改主网络的LSTM状态。现在各自独立。

### 3. 目标网络软更新（每步）

```
tau = 5e-3
QMainNet.softUpdateTo(QTargetNet, tau)  // 每轮学习都执行
```

### 4. 每序列重置LSTM状态

每次训练一个序列前调用 `lstm->reset()` 和 `lstmTarget->reset()`，确保每个序列独立训练，序列内保持时序依赖。

### 5. 降低学习率

`RMSProp(learningRate * 0.5, 0.9, 0)` — LSTM梯度的方差比MLP大，降低学习率有助于稳定训练。

---

## 优化前后对比

| 项目 | 原始实现 | 优化后 |
|------|---------|--------|
| 训练数据采样 | 随机抽取 | **episode内连续子序列** |
| 时序依赖 | 被破坏 | **保留BPTT** |
| 目标网络 | 共用LSTM实例 | **独立LSTM实例** |
| 目标更新频率 | 每256步软更新 | **每步软更新** |
| 双网络Q值 | Double DQN（正确） | Double DQN（保留） |
| LSTM状态管理 | 混乱 | 序列独立为主，推理时保留 |
| 学习率 | 完整learningRate | **减半**（更稳定） |
| 经验回放缩减 | 删25% | **删25%**（保留） |
| seqEnds追踪 | ❌ 无 | **有**（episode边界检测） |

## 修改文件

- **rl/qlstm.h**: 新增 `experienceReplaySeq()`、`lstmTarget`、`seqEnds`、`currentSeqId`
- **rl/qlstm.cpp**: 完全重写 `experienceReplay()` → `experienceReplaySeq()`，重写 `learn()`
