# DRPG（策略梯度+LSTM）问题分析报告

## 概述

**DRPG**（Deep Recurrent Policy Gradient）是一种结合 LSTM 和策略梯度（REINFORCE）的算法。其网络架构为：

```
LSTM → LayerNorm<Sigmoid, Post> → Softmax
```

训练流程（`rl/drpg.cpp:reinforce()`）：
1. 计算折扣奖励 `G_t = Σγⁱ·r_{t+i}`
2. 重置 LSTM 状态
3. 对轨迹中每个时间步：forward → 计算 Loss → backward → gradient
4. 调用优化器更新参数（RMSProp）

---

## 🔴 严重错误

### Bug 1: LSTM 梯度传播链完全断裂（架构级严重错误）

**本质**：`Net::backward()` 无法将误差传播到 LSTM 内部，LSTM 的参数永远不会更新。

**原因分析**：

DRPG 的网络结构为 `layers = [LSTM, LayerNorm, Softmax]`。

在 `net.hpp` 的 `Net::backward()` 方法中，反向传播逻辑分几个条件分支：

```cpp
void backward(const Tensor &loss)
{
    layers[outputIndex]->e = loss;  // 设置Softmax的e = dLoss
    for (int i = layers.size() - 1; i > 0; i--) {
        if (layers[i-1]是CONV && layers[i]是FC) {
            // ... 卷积特殊处理
        } else if (layers[i-1]是LSTM && layers[i]是FC) {
            // 仅当LSTM后紧跟FC时才调用 cacheError()
            layers[i]->backward(e);
            layers[i-1]->cacheError(e);  // ← 只有这个分支调用cacheError
        } else if (layers[i-1]是Attention/MHA/Concat && layers[i]是FC) {
            // ... attention特殊处理
        } else {
            // DRPG进入此分支
            layers[i]->backward(layers[i-1]->e);
        }
    }
    // 最后对第一层调用单参数 backward
    Tensor inputGrad(layers[0]->o.totalSize, 1);
    inputGrad.zero();
    layers[0]->backward(inputGrad);  // ← 调用的是iLayer默认实现（空函数）
}
```

#### 传播路径分析：

| 步骤 | 操作 | 结果 |
|------|------|------|
| 1 | `layers[2]->e = dLoss` | Softmax 的 e 被设置 ✓ |
| 2 | `layers[2]->backward(layers[1]->e)` | Softmax 将梯度经 `W^T` 反传到 LayerNorm 的 e ✓ |
| 3 | `layers[1]->backward(layers[0]->e)` | LayerNorm 计算输入梯度并写入 LSTM 的 e ✓ |
| 4 | `layers[0]->backward(inputGrad)` | 调用 `LSTM::backward(Tensor&)`（继承自 iLayer 的空实现）❌ |

#### 核心问题：

1. **条件分支不匹配**：LSTM 后跟的是 LayerNorm（不是 FC），所以 LSTM+FC 分支不满足，`cacheError()` 从未被调用
2. **LSTM 未重写单参数 backward**：LSTM 只有 `backward(vector<Tensor>&, vector<Tensor>&)` 用于 BPTT，没有重写 `backward(Tensor&)`
3. **cacheE 永远为空**：没有代码调用 `cacheError()`，`cacheE` 保持空
4. **BPTT 不执行**：在 `LSTM::RMSProp()` 中调用 `backward(cacheX, cacheE)`，但 `cacheE` 为空，BPTT 不产生任何梯度

**后果**：
- LSTM 的所有参数（`wi, ui, bi, wf, uf, bf, wg, ug, bg, wo, uo, bo, w, b`）梯度永远为 0
- LSTM 实际上退化成一个固定的非线性变换
- **仅最后的 LayerNorm 和 Softmax 层能学习**

---

### Bug 2: gradient() 在循环中被多次调用，梯度累加策略与 LSTM 不兼容

在 `reinforce()` 中：
```cpp
for (std::size_t t = 0; t < x.size(); t++) {
    Tensor &out = policyNet.forward(x[t].state, false);
    Tensor dLoss = Loss::CrossEntropy::df(out, target);
    policyNet.backward(dLoss);     // 每个时间步调用
    policyNet.gradient(x[t].state, dLoss);  // 每个时间步调用
}
```

**对普通层（LayerNorm, Softmax）**：`gradient()` 正确累加参数梯度 ✓
**对 LSTM**：`gradient()` 继承自空实现，不累加梯度。梯度在 `RMSProp()` 中的 BPTT 才计算 ❌

但问题在于 `backward()` 也被调用了 x.size() 次：
- 每次调用 `Net::backward(dLoss)` 都会重新设置各层的 `e`
- `layers[0]->e`（LSTM 的 e）被 LayerNorm 的 backward 重复覆盖
- 但 LSTM 并不使用 `e` 成员变量进行梯度计算

这虽然不直接导致错误（因为 LSTM 不使用 `e`），但设计上存在清晰的**架构理解偏差**：编写者假设 LSTM 能像 FC 一样通过 `gradient()` 累加梯度，但实际上 LSTM 的梯度机制完全不同。

---

### Bug 3: LSTM 输出层 tanh 导数缺失（与 `docs/lstm_gradient_analysis.md` 记录的 Bug 1 相同）

文件 `rl/lstm.cpp` 第 137 行：

```cpp
// ❌ 使用原始 E，缺少 tanh 导数
Tensor::MM::kijk(delta.h, w, outputError);  // 第142行（已修复？）
```

等等，让我重新检查代码... 实际代码：

```cpp
// 第138-141行: 构造了 outputError
Tensor outputError(hiddenDim, 1);
for (std::size_t i = 0; i < outputError.size(); i++) {
    outputError[i] = E[i] * Tanh::df(states[t].y[i]);
}
// 第142行: 使用了 outputError
Tensor::MM::kijk(delta.h, w, outputError);
```

**已经修复了！** `docs/lstm_gradient_analysis.md` 记录的 Bug 1 在当前代码中已被修正。
第 138-141 行构造 `outputError` 并在第 142 行使用，而非原始 E。

但 `docs/lstm_gradient_analysis.md` 中说的是「第137行使用原始E」，代码行号可能已有变化。

---

## 🟠 算法逻辑错误

### Bug 4: 策略梯度目标函数错误

DPG 和 DRPG 使用相同的策略梯度更新方式：

```cpp
Tensor target = prob;
target[k] = prob[k] * (discountedReward[t] - u);  // u = discountedReward.mean()
Tensor dLoss = Loss::CrossEntropy::df(out, target);
policyNet.backward(dLoss);
```

#### 问题分析：

**标准 REINFORCE 策略梯度**：
```
∇J(θ) = ∇log π(a|s) · (G_t - baseline)
```

**当前实现**：
- 目标是修改概率分布 `target[k] = p[k] · (G_t - u)`，然后通过 CrossEntropy Loss 计算梯度
- CrossEntropy Loss: `L = -Σ target[i] · log(out[i])`
- 梯度: `∂L/∂out[k] = -target[k]/out[k]`（其他维度 = 0）

但这里 `out` 是 softmax 输出，不是 logits。

**标准实现应该是**：
```
dLoss 应该直接是 -∇log π(a|s)·A = -A/π(a|s) 作用于 selected action 维度
或者使用 log_prob * advantage 作为 loss
```

当前方法通过修改概率目标来间接实现策略梯度，**数学上不等价于标准 REINFORCE**。这不是标准实现。

实际上，这个目标从数值上看：
- 如果 `G_t - u > 0`，`target[k] > prob[k]`，CrossEntropy Loss 会推动网络增加 `prob[k]`
- 如果 `G_t - u < 0`，`target[k] < prob[k]`，Loss 会推动网络减小 `prob[k]`

这确实实现了策略梯度的直观目标（增加好动作的概率，减少坏动作的概率），但**不是标准的 ∇logπ·A 形式**，可能导致梯度方向和数值范围不精确。

---

### Bug 5: alpha（温度参数）梯度计算可能错误

```cpp
alpha.g[k] += (RL::entropy(prob[k]) - entropy0) * alpha[k];
```

这里的 `RL::entropy(p)` 定义是：
```cpp
inline float entropy(float p)
{
    return -p * std::log(p);
}
```

所以 `entropy(prob[k])` 是单个动作概率的熵，而不是整个分布的熵。

#### 问题：
- 标准用法应该使用分布熵（整个 softmax 输出的熵），但这里只用了选中动作的概率
- 这可能导致温度参数更新方向错误

---

### Bug 6: 未使用的变量和方法

```cpp
// drpg.h 中声明但从未使用：
Tensor h;
Tensor c;
float learningRate;  // 声明为成员变量但只在参数中传递
```

`h` 和 `c` 在以下情况被使用：
```cpp
// action() 中：
lstm->h = h;
lstm->c = c;

// reinforce() 中：
h = lstm->h;
c = lstm->c;
```

这些赋值实际上**没有被使用在后续计算中**，因为 `reinforce()` 开头就调用了 `lstm->reset()`，然后从头开始运行序列。`h` 和 `c` 的保存/恢复没有实际效果。

实际上这是个设计困惑：`h` 和 `c` 似乎意图在多次训练之间保持 LSTM 状态，但 `reset()` 又将它们清零了。

---

## 🟡 架构设计问题

### Bug 7: LSTM 后接 LayerNorm 的计算数值不稳定

DRPG 的架构是 `LSTM → LayerNorm<Sigmoid, Post> → Softmax`。

LSTM 输出层使用 `tanh` 激活，输出在 (-1, 1) 范围。然后经过 `LayerNorm<Sigmoid, Post>`：

- **Post-LayerNorm**：先线性变换再激活，即 `Sigmoid(W·x + b)`
- LayerNorm 输入是 (-1, 1) 范围，经线性变换后范围扩大，再经 Sigmoid 压缩到 (0, 1)
- Softmax 输入是 Sigmoid 输出

这个设计存在潜在数值不稳定性：
1. LSTM 输出范围 (-1, 1) 经过 LayerNorm 的线性层可扩展到较大范围
2. 但后续 Softmax 直接跟在后面，中间没有数值约束

### Bug 8: BPTT 梯度爆炸/消失风险未被处理

LSTM 的梯度通过 BPTT 传播，但代码中：
- 没有梯度裁剪（除优化器中的 clipGrad 外）
- 没有梯度范数监控
- LSTM 隐藏层维度与输入/输出维度的比例未进行任何数学分析

在 RL 环境中，奖励值可能变化很大，G_t 的计算可能产生大数值，这些都被直接用于梯度计算，增加了 BPTT 不稳定风险。

---

## 🔵 与 DPG 相同的已知问题

### Bug 9: 未存储原始奖励（与 PPO 相同问题）

在 `reinforce()` 中，轨迹的 `reward` 字段没有被修改，所以折扣奖励计算使用的还是原始奖励。但与 PPO 不同的是，DRPG 没有用折扣奖励覆盖原始奖励，所以不会出现「双倍计数未来奖励」的问题。

### Bug 10: CrossEntropy Loss 的 MSE 放大系数

`loss.h` 中的 MSE loss 使用 `2*d*d` 替代 `d*d`，梯度被放大 2 倍。CrossEntropy Loss 是标准实现，不受此影响。

---

## 问题严重程度汇总

| 编号 | 问题描述 | 严重程度 | 影响范围 |
|------|---------|---------|---------|
| 1 | LSTM 梯度传播链完全断裂（cacheError 未被调用） | **🔴 致命** | LSTM 参数永不更新 |
| 2 | gradient() 循环调用与 LSTM 梯度机制不兼容 | **🟠 高** | 训练效率低下，梯度计算冗余 |
| 3 | LSTM 输出层 tanh 导数缺失 → **已修复** | ~~🟠 中~~ | ✅ 已在当前代码中修复 |
| 4 | 策略梯度目标函数非标准形式 | **🟠 中** | 梯度方向和数值不精确 |
| 5 | alpha 温度参数梯度使用单动作熵 | **🟡 低** | 温度调整可能方向错误 |
| 6 | 未使用的 h/c 状态变量 | **🟡 低** | 代码冗余，无功能影响 |
| 7 | LSTM+LayerNorm 数值稳定性 | **🟡 低** | 潜在训练不稳定 |
| 8 | BPTT 梯度爆炸风险未处理 | **🟡 低** | 长期训练可能发散 |

---

## 修复建议

### 修复 1: 修复 LSTM 梯度传播链（最高优先级）

**方案 A**（推荐）：在 `Net::backward()` 中增加对所有 LSTM 类型的处理，无论其后跟随什么层：

```cpp
// net.hpp backward() 中增加分支
} else if (layers[i - 1]->type == iLayer::LAYER_LSTM) {
    // 如果 LSTM 后不是 FC，也需要收集梯度
    Tensor e(layers[i - 1]->o.totalSize, 1);
    layers[i]->backward(e);
    layers[i - 1]->cacheError(e);
} else if (...) {
```

**方案 B**：在 LSTM 中实现单参数 `backward(Tensor&)` 方法，使得 `Net::backward()` 最后的 `layers[0]->backward(inputGrad)` 能起效。

### 修复 2: 修正策略梯度目标函数

改用标准的 REINFORCE 形式：
```cpp
// 使用 log_prob * advantage
float advantage = discountedReward[t] - u;
// 对 softmax 输出计算梯度: dLoss = -advantage * (one_hot(k) - prob)
// 或通过 CrossEntropy: target = one_hot(k), weight = advantage
Tensor target(actionDim, 1);
target.zero();
target[k] = 1;  // one-hot
// 通过 CrossEntropy 实现: L = -advantage * Σ target_i * log(prob_i)
// 但标准的 CrossEntropy df 不包含 advantage 权重
// 更直接的实现：手动构造 dLoss
Tensor dLoss = prob;
dLoss[k] -= 1;  // d(softmax)/d(logit) 近似
// 或更简单的：实现 WeightedCrossEntropy
```

### 修复 3: 简化网络结构，避免不必要的 LayerNorm

将 DRPG 网络改为：
```cpp
LSTM → Softmax
```
去掉中间的 LayerNorm，减少梯度传播路径的复杂度。

### 修复 4: 添加梯度裁剪和监控

在 LSTM 的 BPTT 中添加梯度范数裁剪，防止梯度爆炸。

---

## 结论

DRPG 实现存在最严重的问题是 **LSTM 梯度传播链断裂**（Bug 1），导致 LSTM 参数实际上从未更新，算法退化为仅更新最后两层（LayerNorm + Softmax）。这与原始设计意图严重不符，使得引入 LSTM 带来的时序建模能力完全失效。

策略梯度目标函数的非标准形式（Bug 4）进一步降低了训练效率和梯度精确性。

这两个问题叠加，导致 DRPG 在当前实现下不可能达到设计的性能预期。
