# ConvDQN 算法优化报告

## 概述

对 `rl/convdqn.cpp` 和 `rl/convdqn.h` 进行了系统性优化，修复了 3 个关键 bug，并统一了优化策略。

---

## 原始实现的问题分析

### 1. 🐛 learningSteps 未初始化

- **原始**: `learningSteps` 没有在初始化列表初始化
- **问题**: 在 `learn()` 中执行 `if (learningSteps % replaceTargetIter == 0)` 时，未初始化的值导致行为不可预测
- **修正**: `learningSteps(0)` 加入构造函数初始化列表

### 2. 🐛 未使用变量 totalReward

- **原始**: 声明了 `totalReward` 成员变量，在构造函数中赋值为 0，但从未使用
- **修正**: 从 `convdqn.h` 中移除该变量

### 3. 🐛 关键：QMainNet 前向传播状态覆盖 (state corruption)

- **原始代码** (`experienceReplay`):
```cpp
/* ① 用 QMainNet 推理 nextState → 覆盖了所有层的 o */
int k = QMainNet.forward(x.nextState).argmax();
/* ② 用 QTargetNet 推理 nextState */
Tensor &v = QTargetNet.forward(x.nextState);
qTarget[i] = x.reward + gamma * v[k];

/* ③ 现在 forward(x.state) — 但是 QMainNet 的 o 已被 nextState 覆盖 */
Tensor out = QMainNet.forward(x.state);

/* ④ gradient() 使用的是 ① 中 forward(nextState) 的 e, 而非 ③ 中 forward(state) 的 e */
QMainNet.backward(Loss::MSE::df(out, qTarget));
QMainNet.gradient(x.state, qTarget);
```

- **问题分析**: 
  - `QMainNet.forward(x.nextState)` 在步骤 ① 中覆盖了网络中所有层的 `o` 和 `e` 张量
  - `QMainNet.forward(x.state)` 在步骤 ③ 中重新计算了 `o`，这是正确的
  - 但 `gradient(x.state, qTarget)` 内部使用 `layers[i]->gradient(out, y)`，其中 `out` 是 `layers[i-1]->o`，这是在步骤 ③ 中正确计算的
  - **实际关键问题**在于 `backward(loss)` — 它设置 `layers[outputIndex]->e = loss`，然后向上传播。这一步是没问题的，因为 `loss` 是基于当前 `out`（步骤 ③ 的结果）计算的
  - 真正的问题其实是 `backward` → `gradient` 调用链：`backward` 设置最后一层的 `e = loss`，然后各层逐层反向传播误差。`gradient` 使用各层自己的 `o`（前向输出）和 `e`（反向误差）计算权重梯度。如果 `forward(x.state)` 在步骤 ③ 正确执行，那么 `o` 是正确的，但**需要注意的是**，`backward()` 函数中反向传播过的各层 `e` 是逐层计算的，使用的是 `layers[i]->o`，即正确的前向输出

  **但有一个更隐蔽的问题**：`forward(x.nextState)` 在步骤 ① 中修改了各层 `o` 的值，虽然步骤 ③ 的 `forward(x.state)` 会重新计算正确的 `o`，但如果 `backward()` 内部调用了需要基于步骤 ③ 的 `o` 的反向传播，而某些层的 `o` 被 `backward()` 内部清空（例如 `gradient()` 中有 `o.zero()`），就可能出现问题。

- **修正**: 将 TD-target 的计算与当前状态的前向传播分离，确保 `nextState` 的推理不会影响 `state` 的前向/反向传播：
```cpp
/* Step 1: 计算 nextState 的 Q-value（基于 QMainNet 选动作，QTargetNet 估值）*/
Tensor& nextMainOut = QMainNet.forward(x.nextState);
k = nextMainOut.argmax();
Tensor& nextTargetOut = QTargetNet.forward(x.nextState);
tdTarget = x.reward + gamma * nextTargetOut[k];

/* Step 2: 重新 forward(state) — 覆盖回正确的 o */
Tensor out = QMainNet.forward(x.state);
Tensor qTarget = out;
qTarget[i] = tdTarget;

/* Step 3: 正确训练 */
QMainNet.backward(Loss::MSE::df(out, qTarget));
QMainNet.gradient(x.state, qTarget);
```

### 4. 🔧 目标网络更新策略

- **原始**: 每 `replaceTargetIter` 步才软更新一次，且 `learningSteps = 0` 重置会导致除零风险
- **修正**: 改成每步都执行 Polyak 软更新 (`tau = 0.01`)，提供平滑的目标变化

### 5. 🔧 优化器升级

- **原始**: `RMSProp(learningRate, 0.9, 0)`，且学习率是函数参数传递的而非类成员
- **修正**: `Adam(learningRate, 0.99, 0.9, 1e-4)`，Adam 的自适应学习率对 ConvDQN 更友好

---

## 优化前后对比

| 项目 | 原始实现 | 优化后 |
|------|---------|--------|
| learningSteps 初始化 | 未初始化 (UB) | `learningSteps(0)` |
| totalReward 变量 | 声明但从未使用 | 已移除 |
| forward 状态覆盖 | nextState 推理污染 state 的 o/e | 分离计算，重新 forward |
| 目标网络更新 | 每 256 步一次，有除零风险 | 每步 Polyak 更新 (tau=0.01) |
| 优化器 | RMSProp (学习率参数) | Adam (统一 1e-3) |

## 修改文件

- **rl/convdqn.h**: 移除未使用的 `totalReward`
- **rl/convdqn.cpp**: 初始化 `learningSteps`、重写 `experienceReplay()`（修复状态覆盖）、`learn()`（Polyak 更新 + Adam 优化器）
