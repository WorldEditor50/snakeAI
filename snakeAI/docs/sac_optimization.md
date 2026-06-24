# SAC (Soft Actor-Critic) 算法优化报告

## 概述

对 `rl/sac.cpp` 和 `rl/sac.h` 进行了系统性优化，修复了多个与标准SAC论文不符的关键实现问题，显著提升了算法的稳定性和收敛性能。

## 原始实现的问题分析

### 1. 目标熵设定错误
- **原始**: `entropy0 = RL::entropy(0.1)` = -0.1*log(0.1) ≈ 0.23
- **问题**: 固定值且过小，无法随动作空间动态调整
- **修正**: `entropy0 = -float(actionDim)`，即动作空间维度的负值（标准SAC启发式）

### 2. Q值目标计算 (Clipped Double Q-learning)
- **原始**: 使用所有critics的**平均值**: `q /= max_qnet_num`
- **问题**: 平均值无法有效缓解Q值高估问题
- **修正**: 使用**最小值**: `q_min = min(q_min, qi[j])`

### 3. V(s') 计算方式
- **原始**: 使用argmax确定动作，仅考虑单个动作的Q值
- **问题**: 忽略了策略的随机性，学习信号不连续
- **修正**: 使用**期望形式**: `V(s') = Σ π(a'|s') * (minQ(s',a') - α*log(π(a'|s')))`

### 4. 策略梯度损失函数
- **原始**: `loss[i] = p[i] - p[i]*(qVal - alphaLogP)`
- **问题**: 损失计算逻辑有误，与SAC理论公式不符
- **修正**: `err[i] = α*log(π(a|s)) + α - Q(s,a)`

### 5. 温度参数α更新
- **原始**: 逐元素熵梯度 `alpha.g[i] += (RL::entropy(prob[i]) - entropy0)*alpha[i]`
- **问题**: 使用了动作概率而非动作维度上的熵，且乘了α值导致梯度扭曲
- **修正**: 整体熵梯度 `∇α = H₀ - H`，其中 `H = -Σ π(a|s)*log(π(a|s))`

### 6. α稳定性
- **原始**: 无梯度裁剪和范围限制
- **问题**: α可能发散到极端值
- **修正**: 梯度L2归一化 + 范围限制 `clamp(0.01, 20.0)`

### 7. 目标网络更新
- **原始**: 每个critic使用不同系数 `(i + 1)*1e-4`
- **问题**: 导致目标网络更新步调不一致
- **修正**: 统一使用Polyak系数 `tau = 5e-3`

## 优化前后对比

| 项目 | 原始实现 | 优化后 |
|------|---------|--------|
| 目标熵 | 固定值0.23 | -actionDim (自适应) |
| Q目标聚合 | 平均值 | 最小值 (Clipped Double Q) |
| V(s')计算 | argmax离散 | 期望形式 |
| 策略梯度 | p - p*(Q - αlogπ) | αlogπ + α - Q |
| α更新 | 逐元素错误梯度 | 全局熵梯度 |
| α稳定性 | 无保护 | 梯度裁剪 + 值域限制 |
| Polyak系数 | 各critic不同 | 统一tau=5e-3 |

## 关键公式

### Critic Loss
$$J_Q = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \frac{1}{2} (Q(s,a) - (r + \gamma V(s')))^2 \right]$$

### Value Target
$$V(s') = \sum_{a'} \pi(a'|s') \left( \min_{i=1,\dots,N} Q_i(s',a') - \alpha \log \pi(a'|s') \right)$$

### Policy Loss
$$J_\pi = \mathbb{E}_{s \sim D} \left[ \sum_a \pi(a|s) \left( \alpha \log \pi(a|s) - \min_{i=1,\dots,N} Q_i(s,a) \right) \right]$$

### Temperature Loss
$$J_\alpha = \mathbb{E}_{a \sim \pi} \left[ -\alpha \left( H - H_0 \right) \right]$$
其中 $H = -\sum_a \pi(a|s) \log \pi(a|s)$，$H_0 = -\text{actionDim}$

## GumbelMax策略问题分析

### 原始实现的问题
```cpp
RL::Tensor &RL::SAC::gumbelMax(const RL::Tensor &state)
{
    Tensor& out = actor.forward(state);  // actor末层是Softmax，out已经是概率
    return RL::gumbelSoftmax(out, alpha.val);  // 在概率上加Gumbel噪声后再次Softmax！
}
```

### 两个严重错误

1. **双重Softmax**: actor网络末层已是 `Layer<Softmax>`，`out` 输出的是概率分布。`gumbelSoftmax` 应在**logits**上加Gumbel噪声再Softmax，但在概率上重复Softmax会扭曲分布。

2. **概念混淆**: α是SAC的**熵正则化系数**（控制奖励与熵的平衡），被误用作Gumbel-Softmax的**松弛温度**τ。这两个参数在数学上意义完全不同。

### 修复方案

在 `agent.cpp` 中使用 `Random::categorical` 直接从actor输出的概率分布中采样：

```cpp
// 之前（错误）：
RL::Tensor& a = sac.gumbelMax(state);
int k = a.argmax();

// 之后（正确）：
const RL::Tensor& prob = sac.action(state);
int k = RL::Random::categorical(prob);
```

`Random::categorical` 根据概率分布进行随机采样，既提供了探索性，又保持了正确的概率语义。argmax则用于评估阶段选择最优动作。

## 修改文件

- **rl/sac.cpp**: 完整重写了 `experienceReplay()` 和 `learn()` 方法
- **rl/sac.h**: 修改 `experienceReplay` 签名（移除未使用的 `beta` 参数）
- **agent.cpp**: sacAction中使用 `Random::categorical` 替代 `gumbelMax`


