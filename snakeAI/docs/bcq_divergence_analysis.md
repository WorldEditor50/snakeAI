# BCQ 算法输出发散原因深度分析

## 概述

本报告深入分析 `rl/bcq.cpp` 中 BCQ (Batch-Constrained Q-learning) 算法输出发散的根本原因。代码经过 `docs/bcq_optimization.md` 中描述的优化后，仍然可能存在发散问题。通过对完整训练链路（VAE → Actor → Critic → 优化器）的逐层分析，发现了 **5 个致命缺陷** 和 **3 个严重稳定性问题**。

---

## 🚨 致命缺陷（Divergence Guaranteed）

### 1️⃣ 【致命】Critic 输出 Sigmoid 无法表示 TD-target 范围

| 项 | 值 |
|:---|:---|
| Critic 输出激活函数 | `Layer<Sigmoid>::_` → 输出范围 **[0, 1]** |
| 奖励范围 | `reward0()`: [-1.5, 1.5] |
| 折扣因子 γ | 0.99 |
| TD-target = r + γ·Q' | 最小值: -1.5 + 0.99×0 = **-1.5** |
| | 最大值: 1.5 + 0.99×1 = **2.49** |

**Code Location**: `rl/bcq.cpp:26`

```cpp
critics[i] = Net(Layer<Tanh>::_(featureDim, hiddenDim, true, true),
                 TanhNorm<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
                 Layer<Sigmoid>::_(hiddenDim, actionDim, true, true));  // ← [0,1] bounded!
```

**发散机制**:

```
TD-target ∈ [-1.5, 2.49]  ⊈  Sigmoid-output ∈ [0, 1]

                  ↓
Critic 永远无法拟合 TD-target
                  ↓
每次更新: loss > 0, 梯度始终指向增大 Q 的方向
                  ↓
权重不断增大 → 激活值进入 Sigmoid 饱和区
                  ↓
梯度消失前的持续发散 → Q 值不稳定
                  ↓
错误的 Q 值用于 Actor 更新 → Actor 崩溃
```

**验证**: 假设最优策略下 Q* 约为 1.0 (单步)+ 0.99×1.0 + ... = 收敛值可能远大于 1。例如三步最优路径: 1.5 + 0.99×1.5 + 0.99²×1.5 ≈ 4.47。Critic 受 Sigmoid 限制永远无法输出正确值。

### 2️⃣ 【致命】VAE Decoder Sigmoid 无法重建负值状态

**Code Location**: `rl/vae.hpp:36-38`

```cpp
decoder = Net(Layer<Tanh>::_(zDim, hiddenDim, true, true),
              Layer<Sigmoid>::_(hiddenDim, hiddenDim, true, true),
              Layer<Sigmoid>::_(hiddenDim, inputDim, true, true));  // ← [0,1] output
```

**问题**: VAE 输入为 `concat(state, action)` = 8-dim，其中:
- `state` 值范围: [-1, 1] (归一化后的位置坐标)
- `action` 值范围: {0, 1} (one-hot)

**decoder 的 Sigmoid 输出范围 [0, 1] 无法表示负值的 state 分量**。这意味着 VAE 重建损失永远 > 0，VAE 训练永远不收敛，decoder 输出始终是误差累积后的垃圾值。

### 3️⃣ 【致命】`decode(state)` 直接将状态作为潜变量

**Code Location**: `rl/bcq.cpp:41, 78, 111`

```cpp
// 在 action() 中:
Tensor &BCQ::action(const Tensor &state)
{
    return actor.forward(encoder.decode(state));  // state → 直接解码
}

// 在 experienceReplay() 中:
Tensor decoded = encoder.decode(x.nextState);  // nextState → 直接解码
Tensor decoded = encoder.decode(x.state);      // state → 直接解码
```

**对比 VAE 训练时的正确流程**:

```cpp
// VAE::forward() - 训练时:
Tensor& feature = encoder.forward(x);          // x → 编码器
Tensor& u = meanLayer->forward(feature);       // feature → μ
Tensor& std = stdLayer->forward(feature);      // feature → σ
z[i] = u[i] + std[i]*eps[i];                  // 重参数化采样
return decoder.forward(z);                     // z → 解码器

// VAE::decode() - 推理时:
Tensor& decode(const Tensor &zi) {
    return decoder.forward(zi);                // zi → 直接解码（跳过编码器）
}
```

**发散机制**:

```
VAE 训练时:  x → [Encoder] → u,σ → [Sample: z=u+σ·ε] → [Decoder] → x̂
decode() 时: state → [Decoder] ← 状态值不在潜变量 z 的分布空间中！

        潜变量 z 的分布: z ~ N(u, σ²)   (由编码器决定)
       state 的分布: [-1, 1] 均匀      (环境状态)

=> decoder 接收到从未训练过的输入分布 → 输出 ≈ 随机噪声
```

### 4️⃣ 【致命】平均 Q 而非 Clipped Double Q 导致过估计

**Code Location**: `rl/bcq.cpp:84-91`

```cpp
float qAvg = 0;
for (int i = 0; i < max_qnet_num; i++) {
    const Tensor& qi = criticsTarget[i].forward(decoded);
    qAvg += qi[bestAction];
}
qAvg /= max_qnet_num;
float tdTarget = x.done ? x.reward : x.reward + gamma * qAvg;
```

**标准 BCQ / TD3 使用 min(Q₁, Q₂) = Clipped Double Q-learning** 来抑制过估计：

```
标准 Clipped Double Q:  y = r + γ · min(Q₁(s', a'), Q₂(s', a'))
    本实现 (平均 Q):     y = r + γ · (Q₁ + Q₂)/2
```

**为什么 min 重要**: Q-learning 天然存在正向偏差 (max over actions + bootstrapping)。当两个 Critic 的误差正相关时，平均 Q 不抑制偏差，反而放大过估计 → Q 值持续膨胀 → Actor 学到错误的动作偏好 → 策略退化。

### 5️⃣ 【致命】未使用标准 BCQ 的候选动作生成机制

标准 BCQ 的核心机制:
1. VAE 生成候选动作: 采样多个 z ~ N(0,1)，decode 得到候选 (ŝ, â)
2. Actor 对候选动作加扰动/选择: π(s, â) → 修正动作
3. 扰动后的动作与环境交互

**本实现完全跳过了候选生成**:

```cpp
// 本实现:
Tensor &BCQ::action(const Tensor &state) {
    return actor.forward(encoder.decode(state));
    // state → decode(跳过了采样) → 8-dim → actor → 4-dim (动作概率)
}

// 标准 BCQ (离散动作版):
// 采样 N 个 z ~ N(0,I), decode 得到候选动作, actor 扰动, 选最优 Q
```

`mixAction()` 实现了接近标准的结构，但**从未被调用**——`agent.cpp:526` 调用的是 `bcq.action(state)`。

---

## ⚠️ 严重问题（Significant Instability）

### 6️⃣ 状态观察的维度混淆

**Code Location**: `agent.cpp:48-57`

```cpp
void Agent::observe(RL::Tensor& statex, int x, int y, int xt, int yt) {
    float xc = float(env.map.shape[0]) / 2;
    float yc = float(env.map.shape[1]) / 2;
    statex[0] = (x - xc) / xc;   // → [-1, 1]
    statex[1] = (y - yc) / yc;   // → [-1, 1]
    statex[2] = (xt - xc) / xc;  // → [-1, 1]
    statex[3] = (yt - yc) / yc;  // → [-1, 1]
}
```

状态维度为 4。但 VAE 输入是 `concat(state, action)` = **8-dim**。
- state: 4-dim, range [-1, 1]
- action: 4-dim, one-hot {0, 1}
- VAE 输入: 8-dim

**Sigmoid 输出 [0,1] 对 state 分量 [-1,1] 的重建是根本不可能的** —— 这与问题 #2 相同, 但在此加深影响。

### 7️⃣ Actor 训练中 Critic 输入使用 VAE 解码输出而非 concat(state, action)

**Code Location**: `rl/bcq.cpp:111-117`

```cpp
Tensor decoded = encoder.decode(x.state);  // 8-dim
actor.forward(decoded);                     // 8-dim → 4-dim

Tensor q(actionDim, 1);
for (int i = 0; i < max_qnet_num; i++) {
    const Tensor& qi = critics[i].forward(decoded);  // critic(8-dim decoded)!
    // ...
}
```

原始 BCQ 论文中 Critic 的输入是 `concat(state, action)`，但这里传入的是 `decoded` (VAE decoder 的输出 [ŝ, â])。由于 decoder 输出被问题 #2-#3 污染，Critic 的 Q 值评估基于 corrupted feature，进一步加剧发散。

### 8️⃣ 学习率与每步更新可能不匹配

**Code Location**: `rl/bcq.cpp:146-164`

```cpp
for (std::size_t i = 0; i < batchSize; i++) {
    int k = uniform(Random::engine);
    experienceReplay(memories[k]);
}

actor.Adam(lr, 0.99, 0.9, 1e-4);
encoder.Adam(lr, 0.99, 0.9, 1e-4);
for (int i = 0; i < max_qnet_num; i++) {
    critics[i].Adam(lr, 0.99, 0.9, 1e-4);
}
```

- 每步（每次 `bcqAction` 调用）从 0-8192 的缓冲区随机采样 32 个 transition
- 使用 lr=1e-3 进行 Adam 更新
- 每个 transition 中 Actor 和 Critic 都做一次前向+后向+梯度累积
- 这种"每个 transition 独立更新" + "高学习率" 的组合对离线 RL 不稳定

---

## 🔄 发散因果链（Root Cause Chain）

```
问题 #2 (VAE Sigmoid 无法重建负状态)
  → VAE 训练永不收敛
  → decoder 输出始终为噪声
      ↓
问题 #3 (decode(state) 使用状态作潜变量)
  → decoder 输出完全失真
  → encoder 从未参与推理过程
      ↓
问题 #5 (无候选动作生成)
  → Actor 接收的是 VAE decoder 的垃圾输出
  → Actor 学习到错误映射
      ↓
问题 #4 (平均 Q 过估计)
  + 问题 #1 (Sigmoid 限制 Q 值范围)
  → Critic 永远无法拟合 TD-target
  → 梯度指向使权重发散的方向
  → Critic Q 值不稳定 → 错误信号反馈给 Actor
      ↓
问题 #7 (Critic 输入使用 VAE 解码)
  → Critic 基于 corrupted features 评估 Q
  → Q 值与真实动作价值完全无关
      ↓
                        最终发散
```

---

## 🔧 修复建议

| 优先级 | 问题 | 修复 |
|:------:|:----|:-----|
| 🔴 P0 | #1 Critic Sigmoid 范围 | 将 Critic 输出层改为 `Linear` (无激活函数) |
| 🔴 P0 | #2 VAE Decoder Sigmoid 范围 | 将 decoder 输出层改为 `Tanh` (输出范围 [-1,1] 匹配输入) |
| 🔴 P0 | #3 decode(state) 误用 | 添加 `priorDecode()` 从先验采样 z~N(0,1)，或在 decode 前先通过 encoder 编码 |
| 🔴 P0 | #4 平均 Q 过估计 | 改为 `min(Q₁, Q₂)` Clipped Double Q-learning |
| 🔴 P0 | #5 无候选动作生成 | 实现标准 BCQ 采样流程：采样 z ~ N(0,1) → decode → actor 扰动 |
| 🟠 P1 | #7 Critic 输入错误 | Critic 输入统一使用 `concat(state, action)` |
| 🟠 P1 | #8 学习策略 | 降低学习率至 3e-4 |

---

## 附录：标准 BCQ 算法参考

标准离散 BCQ 的核心流程:

```
1. VAE 训练 (重建 concat(s, a)):
   z ~ Encoder(concat(s, a))
   x̂ = Decoder(z)
   Loss = MSE(x̂, concat(s, a)) + β·KL(N(μ,σ)||N(0,1))

2. 动作选择 (推理时):
   重复 N 次:
     z ~ N(0,1)                    # 从先验采样
     (ŝ, â) = Decoder(z)         # 解码得到候选动作
   π(s, â) = Actor(concat(s, â)) # 扰动/选择
   a* = argmax(Q(s, π(s, â)))

3. Critic 训练:
   y = r + γ·min(Q₁'(s', a*), Q₂'(s', a*))
   L = MSE(Q₁(concat(s,a))[a], y) + MSE(Q₂(concat(s,a))[a], y)

4. Target 网络软更新:
   θ' ← τ·θ + (1-τ)·θ'
```

当前实现与标准 BCQ 存在多处根本性差异，这些差异的组合效应导致了输出发散。
