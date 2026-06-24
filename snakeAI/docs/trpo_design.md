# TRPO (Trust Region Policy Optimization) 设计与实现文档

## 概述

TRPO（Trust Region Policy Optimization）是一种基于信赖域的策略梯度算法，通过**自然梯度**方向更新策略，并引入**KL散度约束**来确保每次迭代的策略更新步长不会太大，从而提高训练的稳定性和单调性。

本实现基于框架中已有的神经网络组件（Net、Layer、Tensor等）构建，与PPO共享相似的网络架构，但使用不同的优化策略。

---

## 一、算法原理

### 1.1 核心思想

TRPO的优化目标是在限制新旧策略差异的前提下最大化替代目标函数：

```
maximize      E[π_new(a|s) / π_old(a|s) · A(s,a)]
subject to   E[KL(π_old(·|s) || π_new(·|s))] ≤ δ
```

其中δ为最大KL散度约束（`maxKL`），A(s,a)为优势函数。

### 1.2 自然梯度法

TRPO使用**自然梯度**而非标准梯度方向，通过Fisher信息矩阵（FIM）Hessian矩阵对梯度进行二次缩放：

```
Δθ ∝ H^(-1) · g

其中:
  g = ∇_θ L(θ)          ← 标准策略梯度
  H = E[∇²_θ KL]        ← Fisher信息矩阵（KL散度的Hessian）
```

自然梯度的优势：
- **参数化不敏感**：不受网络参数化方式影响
- **二阶收敛性**：考虑了曲率信息，收敛更快

### 1.3 共轭梯度法（CG）

直接计算 H^(-1) 的逆矩阵代价过高（O(n³)），TRPO使用**共轭梯度法**迭代求解 Hx = g：

```
算法：Conjugate Gradient
输入：g（梯度向量），Hv函数（Hessian-vector product oracle）
输出：x ≈ H^(-1)·g

r₀ = g
p₀ = r₀
for k = 0,1,...:
    α_k = r_kᵀr_k / (p_kᵀ H p_k)
    x_{k+1} = x_k + α_k · p_k
    r_{k+1} = r_k - α_k · H p_k
    if ‖r_{k+1}‖ < ε: break
    β_k = r_{k+1}ᵀr_{k+1} / (r_kᵀr_k)
    p_{k+1} = r_{k+1} + β_k · p_k
```

共轭梯度只需Hv运算（O(n)而非O(n²)），非常适合大规模参数空间。

### 1.4 Hessian-Vector Product（Hv）

Hv通过**有限差分法**近似计算，避免显式构建Hessian矩阵：

```
Hv ≈ (∇_θ KL(π_old || π_θ') - ∇_θ KL(π_old || π_θ)) / ε

其中 θ' = θ + ε·v
```

每次Hv只需要两次梯度计算（扰动前后各一次），复杂度O(n)。

### 1.5 步长缩放与回溯线搜索

共轭梯度得到的方向 x 需要缩放以满足KL约束：

```
stepSize = √(2·δ / xᵀ H x)
Δθ = x · stepSize
```

然后执行回溯线搜索：
```
for α ∈ {1, 0.5, 0.25, 0.125, ...}:
    θ_new = θ_old + α · Δθ
    if KL(π_old || π_new) ≤ 1.5·δ and L(θ_new) ≥ L(θ_old):
        accept
```

---

## 二、架构设计

### 2.1 类结构

```
┌─────────────────────────────────────────┐
│                TRPO                      │
├─────────────────────────────────────────┤
│  Actor Networks:                         │
│    actorP  ← 当前策略（训练中更新）         │
│    actorQ  ← 旧策略（冻结，用于KL约束）     │
│  Critic Network: 状态-动作值函数          │
│  Hyperparameters:                        │
│    gamma    = 0.99   折扣因子             │
│    lmbda    = 0.95   GAE λ参数          │
│    maxKL    = 0.01   KL约束上限          │
│    delta    = 0.01   信赖域半径          │
│  Temperature:                            │
│    alpha    ← Gumbel-Softmax温度参数      │
│    annealing← 退火调度器                  │
└─────────────────────────────────────────┘
```

### 2.2 网络架构

与PPO相同的网络结构：

| 组件 | 架构 | 激活函数 |
|------|------|---------|
| Actor (actorP) | state → 128 → 128 → 128 → 128 → action | Tanh/TanhNorm/Softmax |
| Actor旧策略 (actorQ) | 同上（复制于actorP） | Tanh/TanhNorm/Softmax |
| Critic | state+action → 128 → 128 → 128 → 128 → action | Tanh/TanhNorm/Linear |

### 2.3 与PPO的对比

| 维度 | PPO | TRPO |
|------|-----|------|
| 优化方法 | 一阶梯度 | 二阶自然梯度 |
| KL约束方式 | 自适应KL惩罚β·KL 或 clip(r,1-ε,1+ε) | 硬约束 + 回溯线搜索 |
| 计算复杂度 | O(n) | O(n·CG_iter) |
| Hessian矩阵 | 不需要 | 通过Hv近似 |
| 共轭梯度 | 不需要 | 需要（默认20步） |
| 理论保证 | 无严格单调提升保证 | 有（信赖域保证） |

---

## 三、数学推导

### 3.1 策略梯度

替代目标函数的梯度：

```
L(θ) = (1/N) · Σ_t π_new(a_t|s_t)/π_old(a_t|s_t) · A_t

∇L(θ) = (1/N) · Σ_t A_t · ∇log π_new(a_t|s_t)
       = (1/N) · Σ_t A_t · 1/p_k · ∇p_k
```

其中 actionDim 维度上仅有 argmax 动作 k 非零。

### 3.2 KL散度梯度

KL散度：D_KL(q || p) = Σ_i q_i · log(q_i / p_i)

对于动作概率分布：

```
∂D_KL/∂p_i = -q_i / p_i
```

### 3.3 广义优势估计（GAE）

使用TD误差的指数加权平均，平衡偏差与方差：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

### 3.4 标量优势 vs 向量值函数

TRPO（本实现）使用**向量值critic**：Critic网络输出 actionDim 维向量 V(s,a)，表示每个离散动作的值函数。优势计算方式为：

```
V(s_t) = critic(s_t, a_t)    → 标量 v = q_target[k]
A_t = tdTarget_t - v
```

---

## 四、关键实现细节

### 4.1 训练流程（learn()）

```
1.  actorP.copyTo(actorQ)                    ← 冻结旧策略
2.  Critic更新（MSE损失 + TD目标）
3.  GAE优势估计（λ=0.95）
4.  计算策略梯度 g
5.  共轭梯度法求解 Hx = g（20步）
6.  步长缩放：s = √(2·δ / xᵀHx)
7.  回溯线搜索（α∈{1, 0.5, 0.25, ...}, max 10步）
8.  温度参数退火
```

### 4.2 参数扁平化

由于框架中 Net 的参数存储在各层的权重矩阵中，TRPO需要将所有参数展平为一维向量，以便执行CG、Hv等线性代数运算：

```
flatParams()    → [W1, b1, W2, b2, ..., W5, b5]
setFlatParams() → 写回网络各层
flatGrad()      → 从各层梯度缓存收集
```

`ACTOR_LAYERS = 5` 对应构造器中5个层的数量。

### 4.3 Hessian-Vector Product 精度

使用有限差分法时，ε的选择影响数值精度：

- 单样本版 `hessain()`：ε = 1e-5
- 整轨迹版 `hessianVectorProduct()`：ε = 1e-5

ε过大会引入偏差，过小会导致数值不稳定。

### 4.4 共轭梯度收敛判定

CG迭代在以下任一条件满足时提前终止：
1. 残差 ‖r‖ < 1e-10（已经收敛）
2. pᵀHp ≤ 1e-10（出现负曲率或数值问题）
3. 达到最大迭代次数（默认20次）

### 4.5 回溯线搜索保证

线搜索确保同时满足两个条件：
1. **KL约束**：KL(π_old || π_new) ≤ 1.5·maxKL（允许轻微超界）
2. **替代目标不下降**：L(θ_new) ≥ L(θ_old) - 1e-6（允许极小下降）

如果10次回溯后仍未满足，则恢复原始参数（不更新）。

---

## 五、使用示例

### 5.1 创建TRPO智能体

```cpp
#include "trpo.h"

// 状态维度=16, 隐藏层=128, 动作维度=4
RL::TRPO agent(16, 128, 4);

// 采样动作（使用Gumbel-Softmax探索）
RL::Tensor action = agent.gumbelMax(state);

// 执行动作并收集轨迹
// ...

// 训练（传入整条轨迹）
agent.learn(trajectory, 0.001);

// 保存模型
agent.save("actor.para", "critic.para");

// 加载模型
agent.load("actor.para", "critic.para");
```

### 5.2 与PPO的切换

```cpp
// PPO版本
RL::PPO ppo(16, 128, 4);
ppo.learnWithClipObjective(trajectory, 0.001);

// TRPO版本（同架构，不同优化策略）
RL::TRPO trpo(16, 128, 4);
trpo.learn(trajectory, 0.001);  // learningRate参数未使用（TRPO自动确定步长）
```

---

## 六、性能分析

### 6.1 计算复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| Critic更新 | O(N·n_actor) | N=轨迹长度，n_actor=动作维度 |
| 策略梯度 | O(N·n_actor) | 单次前向+反向 |
| CG求解 | O(C·(N·n_actor)) | C=CG迭代次数（默认20），每次Hv需2次梯度 |
| 步长缩放 | O(CG_iter·(N·n_actor)) | 额外一次Hv |
| 线搜索 | O(B·(N·n_actor)) | B=回溯次数（最多10） |
| **总计** | **O((C+2)·(N·n_actor))** | 约PPO的20倍（CG开销） |

### 6.2 推荐使用场景

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 小规模策略网络 | TRPO | 二阶优化的优势明显 |
| 大规模策略网络 | PPO | CG迭代过于昂贵 |
| 高精度要求 | TRPO | 信赖域保证收敛稳定性 |
| 快速实验迭代 | PPO | 训练速度快 |

---

## 七、文件清单

| 文件 | 位置 | 说明 |
|------|------|------|
| trpo.h | rl/trpo.h | 类声明（公共接口 + 私有辅助方法） |
| trpo.cpp | rl/trpo.cpp | 完整实现（416行） |
| 本设计文档 | docs/trpo_design.md | 算法原理与实现说明 |

---

## 八、已知限制与改进方向

### 8.1 当前限制

1. **仅支持离散动作**: 基于Softmax输出，连续动作需重新设计
2. **单轨迹更新**: 每次 learn() 处理一条完整轨迹，不支持batch
3. **CG迭代固定**: 20次CG迭代，无自适应终止
4. **无v-trace**: 不支持off-policy校正

### 8.2 可能的改进

1. **Fisher矩阵裁剪**: 防止Hessian奇异值过大
2. **阻尼（Damping）**: 在Hessian对角线添加λI增强数值稳定性
3. **自适应CG步数**: 根据残差下降率动态调整
4. **自然梯度PPO**: 结合PPO的clip目标与自然梯度方向
5. **连续动作支持**: 使用高斯分布策略代替Softmax
