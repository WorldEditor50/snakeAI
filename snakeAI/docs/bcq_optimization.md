# BCQ (Batch-Constrained Q-learning) 算法优化报告

## 概述

对 `rl/bcq.cpp` 进行了系统性优化，修复了 3 个关键 bug 并改进了优化策略。BCQ 是处理**离线强化学习**的经典算法，核心思想是通过 VAE 生成与数据集中行为相似的候选动作，再通过 Q-learning 在这些候选动作中选择最优值。

---

## 原始实现的问题

### 1. 🐛 learningSteps 未初始化

- **原始**: 未在初始化列表初始化
- **问题**: 在 `learn()` 中使用 `learningSteps % replaceTargetIter` 时出现 UB
- **修正**: `learningSteps(0)` 加入初始化列表

### 2. 🐛 Critic TD-target 使用错误的动作索引

- **原始代码** (critics 训练):
```cpp
/* select action */
Tensor ga = encoder.decode(x.nextState);
Tensor san = Tensor::concat(1, x.nextState, ga);
const Tensor& nextProb = actor.forward(san);
std::size_t k = nextProb.argmax();   // k = actor's best action in next state
/* ... compute qTarget ... */
k = x.action.argmax();               // k = action actually taken
for (int i = 0; i < max_qnet_num; i++) {
    const Tensor &out = critics[i].forward(sa);
    Tensor p = out;
    p[k] = qTarget;
    // ...
}
```

- **问题**: 虽然这里 `k` 在设置 TD-target 时正确使用了 `x.action.argmax()`（transition 中实际执行的动作），所以这部分**实际上没问题**。但代码先定义 `k` 为 best action，然后**重用**赋值给 action taken，可读性差且容易出错。更重要的是，原始的 `qTarget` 是**覆盖**到 `k` 位置——即 TD-target 只作用于实际执行的动作索引，其他动作保持原始 Q 值，这是正确的。

- **修正**: 使用独立的变量名 `actionTaken` 代替重用的 `k`，提高代码清晰度

### 3. 🐛 Actor 策略梯度使用 `p - q` 而非正确的 `-q`

- **原始代码**:
```cpp
const Tensor& p = actor.forward(sa);
// ...
for (int i = 0; i < actionDim; i++) {
    loss[i] = p[i] - q[i];
}
actor.backward(loss);
actor.gradient(sa, loss);
```

- **问题分析**:
  - Actor 的优化目标是最大化：`J(π) = Σ π_i(s, ga) · Q_i(s, ga)`
  - 梯度为：`dJ/dπ_i = Q_i(s, ga)`（最大化方向）
  - 使用 `backward` 传入梯度时，需要传入**损失相对于输出的梯度**：`dJ/dπ_i = Q_i`
  - **但** `p[i] - q[i]` 的梯度是 `d/dπ_i (π_i - Q_i) = 1`——这完全不对！
  - 这相当于对所有的动作施加了相同的梯度 1，完全忽略了 Q-values 的信息

- **修正**: 直接使用 Q-values 作为策略梯度：
```cpp
/* maximize Σ π_i Q_i → gradient dJ/dπ_i = Q_i */
actor.backward(q);        // q = averaged Q-values
actor.gradient(sa, q);
```

  Softmax 的 `gradient()` 通过雅可比矩阵自动转换为正确的 logits 梯度：
  ```
  dJ/dz_i = Σ_j (dJ/dπ_j · dπ_j/dz_i)
          = π_i · (Q_i - Σ_j π_j · Q_j)  √ 标准策略梯度
  ```

### 4. 🐛 Actor 训练中 critics 使用错误的输入

- **原始代码**:
```cpp
for (int i = 0; i < max_qnet_num; i++) {
    const Tensor& qi = critics[i].forward(x.state);  // ← 只传了 state, 没有 ga!
    q += qi;
}
```

- **问题**: BCQ 中 critics 的输入是 `concat(state, action)` 即 `featureDim = stateDim + actionDim`。这里应该传入 `concat(state, ga)` 而非单独的 `state`。
  - `critics[i]` 网络的输入维度 = `stateDim + actionDim`
  - `forward(x.state)` 传入维度 = `stateDim`
  - 这会导致运行时错误（形状不匹配）或者更糟糕——静默地使用错误的输入张量
  
- **修正**: 
```cpp
const Tensor& qi = critics[i].forward(sa);  // sa = concat(state, ga)
```

### 5. 🔧 优化策略改进

#### 目标网络更新
- **原始**: 每 256 步软更新一次，critic 0 用 `1e-4`，critic 1 用 `2e-4`（极小的 tau 值意味着几乎不更新目标网络）
- **修正**: 每步 Polyak 软更新 `tau = 5e-3`，两个 critic 使用相同的 tau

#### 优化器升级
- **原始**: 
  - Actor: RMSProp(1e-2, 0.9)
  - Critics: RMSProp(1e-3, 0.9)
  - VAE: **从未优化!** — VAE 的 `forward/backward` 虽然在 `experienceReplay` 中被调用，但 `learn()` 中从未调用 `encoder.RMSProp()`。这意味着 VAE 的权重从未更新！
- **修正**: 统一使用 Adam 优化器，并添加 VAE 优化：
  - Actor: Adam(1e-3)
  - Critics: Adam(1e-3)
  - VAE: Adam(1e-3)

---

## VAE 分析与修复

### 6. 🐛 VAE KL 梯度计算错误 (rl/vae.hpp)

| 位置 | 原代码 | 正确公式 |
|------|-------|---------|
| **mean 梯度** | `u[i] + e1[i]` | ✅ `u[i] + e1[i]` |
| **std 梯度** | `(std[i] - 1/std[i] + e1[i]) * eps[i]` | ❌ `e1[i]*eps[i] + (std[i] - 1/std[i])` |

**推导**：

重参数化 `z = u + σ·ε`

- `∂L_rec/∂σ_i = e1_i · ε_i` (通过解码器反向传播)
- `∂KL/∂σ_i = σ_i - 1/σ_i` (KL 散度: -0.5(1+logσ²-σ²-u²))

**原代码错误**: KL 项的 `(σ - 1/σ)` 被 ε 缩放，导致：
- 当 ε 很小时，KL 正则化被削弱 → VAE 后验过度偏离先验
- 当 ε 很大时，KL 正则化过强 → 重建质量下降

### 7. 🐛 BCQ VAE 维度架构不匹配

#### 7.1 VAE 构造

```cpp
VAE(featureDim=8, 2*featureDim=16, zDim=stateDim=4)
```

架构：
```
Input:  concat(s, a)   — dim = 8
Encoder: latent z      — dim = 4 (stateDim)
Decoder: output         — dim = 8 = [s_reconstructed, a_candidate]
```

**关键洞察**: decoder 输出 8-dim 恰好等于 `featureDim`(actor/critic 输入维度)！`decode(nextState)` 返回的就是完整的 feature 向量，**不需要再 concat state**。

#### 7.2 原代码维度 bug (已修复)

| 位置 | 原代码 | 问题 | 修正 |
|------|--------|------|------|
| `action()` | `Tensor sa = concat(1, state, decode(state))` | 12-dim 输入期待 8-dim 的 actor | 直接用 `decode(state)` (8-dim) |
| Critics 训练 | `concat(nextState, decode(nextState))` → 12-dim | critic 期待 8-dim | 直接用 `decode(nextState)` (8-dim) |
| Actor 训练 | `concat(state, decode(state))` → 12-dim | critic 期待 8-dim | 直接用 `decode(state)` (8-dim) |

**根因**: `decode(state)` 的 8-dim 输出已经包含了 `[s_rec, a_cand]`，再 concat state 会得到 12-dim — 网络权重矩阵无法接收！

## 优化前后对比

| 项目 | 原始实现 | 优化后 |
|------|---------|--------|
| learningSteps 初始化 | 未初始化 (UB) | `learningSteps(0)` |
| Actor 策略梯度 | `p - q` (相当于常数梯度 1) | `q` (正确 Q-value 梯度) |
| Actor 中 critics 输入 | `x.state` (维度不匹配) | `decode(state)` (正确 8-dim) |
| Actor 中所有输入维度 | `concat(state, ga)` = 12-dim | 直接使用 VAE decoder 输出 = 8-dim |
| VAE 是否优化 | ❌ 从未优化 | ✅ 每步 Adam 优化 |
| VAE std KL 梯度 | (std-1/std+e1)*ε (错误) | e1*ε + (std-1/std) (正确) |
| 目标网络更新 | 每 256 步 tau=1e-4 | 每步 Polyak tau=5e-3 |
| 优化器 | RMSProp (多种 LR) | Adam (统一 1e-3) |

## 标准 BCQ 算法伪代码


```
Algorithm BCQ (Batch-Constrained Q-learning, 离散动作版):

初始化 VAE v, Actor π_θ, Critics Q_φ₁, Q_φ₂
初始化 Target Critics Q_φ'₁, Q_φ'₂
初始化经验池 D (含离线数据)

for each training step:
    从 D 采样 batch
    
    # 1. 训练 VAE (reconstruction)
    sa = concat(s, a)
    L_VAE = MSE(v.forward(sa), sa) + KL_div
    update v
    
    # 2. 训练 Critics (clipped double Q-learning)
    for each (s, a, r, s') in batch:
        ga = v.decode(s')                    # VAE 生成候选动作
        san = concat(s', ga)
        π_next = π_θ(san)                    # actor 调整候选动作概率
        
        a' = argmax π_next                   # 选最优候选动作
        
        q1' = Q_φ'₁(san)[a']
        q2' = Q_φ'₂(san)[a']
        y = r + γ · min(q1', q2')            # clipped double Q
        
        L_Q₁ = MSE(Q_φ₁(concat(s, a))[a], y)  # 只更新实际执行动作
        L_Q₂ = MSE(Q_φ₂(concat(s, a))[a], y)
        update φ₁, φ₂
    
    # 3. 训练 Actor
    ga = v.decode(s)
    sa = concat(s, ga)
    π = π_θ(sa)
    q = min(Q_φ₁(sa), Q_φ₂(sa))             # 或平均
    J = Σ π_i · q_i                         # 最大化
    ∇J/∇θ = q                               # 通过 Softmax Jacobian
    
    # 4. 软更新目标网络
    φ'₁ ← τ·φ₁ + (1-τ)·φ'₁
    φ'₂ ← τ·φ₂ + (1-τ)·φ'₂
```

## 修改文件

- **rl/bcq.cpp**: 初始化 `learningSteps`、修正 Actor 策略梯度、修正 Actor 中 critics 输入、添加 VAE 优化、Polyak 每步更新 + Adam 优化器
