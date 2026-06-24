# DDPG (Deep Deterministic Policy Gradient) 算法优化报告

## 概述

对 `rl/ddpg.cpp` 和 `rl/ddpg.h` 进行了系统性优化，修复了多个与标准 DDPG 算法不符的关键实现问题，包括未初始化变量、Critic TD-target 错误、Actor 策略梯度错误、目标网络更新策略不合理等。

---

## 原始实现的问题分析

### 1. 成员变量未初始化

- **原始**: `learningSteps` 未在构造函数初始化列表中初始化
- **问题**: 未初始化的 `learningSteps` 具有不确定的初始值，导致 `if (learningSteps % replaceTargetIter == 0)` 的判断结果不可预测，可能引发除零异常或错误的网络更新时机
- **修正**: 在初始化列表中加入 `learningSteps(0)`

### 2. Critic (Q网络) TD-target 计算错误

- **原始**:
```cpp
/* train critic */
std::size_t i = x.action.argmax();
Tensor ct = criticP.forward(x.state);
if (x.done == true) {
    ct[i] = x.reward;
} else {
    Tensor &aq = actorQ.forward(x.nextState);
    int k = aq.argmax();
    Tensor &cq = criticQ.forward(x.nextState);
    ct[k] = x.reward + gamma*cq[k];
}
```

- **问题1**: 在 `done == false` 的分支中，TD-target 被设置到 `k` 位置（即 `actorQ.argmax()` 对应的动作），而不是 `i` 位置（transition 中实际执行的动作）。这意味着：
  - Critic 学习的实际上是一个**虚构的动作的价值**，而非实际执行的动作的价值
  - `ct[i]` 仍然保留原始 Q 值，导致该动作被错误的 Q-target 监督
  
- **问题2**: 即使是在 `done == true` 的分支，`ct[i] = x.reward` 也只是覆盖了执行动作的 Q 目标，但其他未执行的动作的 Q 目标保持了原有 Q 值没有变化。正确的做法是只对实际执行的动作施加 TD 目标。

- **修正**: 只对 transition 中实际执行的动作构建 TD-target：
```cpp
Tensor qTarget = qCurrent;
qTarget[actionTaken] = tdTarget;
```

### 3. Actor (策略网络) 梯度计算错误

- **原始**:
```cpp
/* train actor: maximize Q(s,π(s)) by pushing π toward argmax Q action */
Tensor &p = actorP.forward(x.state);
Tensor& q = criticP.forward(x.state);
int k = q.argmax();
Tensor dLoss(actionDim, 1);
for (std::size_t i = 0; i < actionDim; i++) {
    if (i == k) {
        dLoss[i] = p[i] - 1;  /* push best action probability toward 1 */
    } else {
        dLoss[i] = p[i];      /* push other actions toward 0 */
    }
}
```

- **问题**: 这是一种**非标准的 hard 策略更新**：
  1. 完全忽略了 Q-values 的相对量级信息。两个动作的 Q 值可能是 0.51 和 0.49，但该方法只关注 argmax，将次优动作的概率硬推为 0
  2. 这本质上是将策略梯度问题简化为了一个分类问题：找到 Q 最大的动作，然后 push policy 向 one-hot 靠近
  3. 无法处理 Q-values 近似相等的情况，导致策略可能在某些状态下过于确定性
  4. 通过 `p[i]` 和 `p[i] - 1` 对 Softmax 输出进行 hard 监督，但 Softmax 层有专门的 Jacobian 计算来处理梯度传播

- **修正**: 使用正确的基于 Q-value 的策略梯度：
```cpp
/* maximize J(π) = Σ π_i(s) * Q_i(s) */
/* minimize -J: gradient w.r.t π_i is -Q_i */
Tensor err(actionDim, 1);
for (std::size_t i = 0; i < actionDim; i++) {
    err[i] = -qValues[i];
}
actorP.backward(err);
actorP.gradient(x.state, err);
```

此时 `-Q_i` 是相对于策略网络**输出层 (Softmax 概率)** 的梯度。
Softmax 层的 `gradient()` 方法通过雅可比矩阵自动将 `dJ/dπ_i` 转换为对 logits 的梯度：
```
dJ/dz_i = Σ_j (dJ/dπ_j · dπ_j/dz_i) = π_i · (Q_i - Σ_j π_j · Q_j)
```
其中 `z_i` 为 Softmax 前的 logits，这实际上是标准策略梯度定理的离散形式。

### 4. 目标网络更新策略不合理

- **原始**: 仅当 `learningSteps % replaceTargetIter == 0` 时才更新目标网络：
```cpp
if (learningSteps % replaceTargetIter == 0) {
    criticP.softUpdateTo(criticQ, 0.01);
    actorP.softUpdateTo(actorQ, 0.01);
    learningSteps = 0;
}
```

- **问题**: 
  1. 标准 DDPG 使用**每步 Polyak 软更新**，而非周期性更新
  2. 软更新的优势在于提供平滑的目标变化，而间隔性更新会导致目标网络突变
  3. `learningSteps = 0` 重置计数器后，`learningSteps % 0` 会触发除零错误

- **修正**: 每步执行 Polyak 软更新：
```cpp
float tau = 5e-3;
criticP.softUpdateTo(criticQ, tau);
actorP.softUpdateTo(actorQ, tau);
```

### 5. 优化器不统一

- **原始**: 
```cpp
actorP.RMSProp(0.01, 0.9, 0.01);
criticP.RMSProp(1e-3);
```

- **问题**: Actor 和 Critic 使用不同的优化器参数（学习率和衰减），且 Actor 使用了额外的 `decay=0.01` 权重衰减

- **修正**: 统一使用 Adam 优化器，有利于更稳定的梯度更新：
```cpp
actorP.Adam(1e-3, 0.99, 0.9, 1e-4);
criticP.Adam(1e-3, 0.99, 0.9, 1e-4);
```

### 6. noiseAction 中 forward 调用后的引用问题

- **原始**:
```cpp
RL::Tensor& RL::DDPG::noiseAction(const Tensor &state)
{
    Tensor& out = actorP.forward(state);
    return noise(out);
}
```

- **问题**: `noise(out)` 对 `out` 进行原地修改，而 `out` 是网络内部存储的张量。`noise()` 函数在第 311 行执行 `x += epsilon; x /= x.max();`，这直接修改了 `actorP` 中最后一层的 `o` 张量。这并非功能错误，但会破坏网络的输出用于后续 `action()` 调用的意图——前向传播的输出本应是策略的概率分布，不应被噪声覆盖。

- **修正**: 使用 `actorP.output()` 获取输出层引用，语义上表明只是读取输出：
```cpp
actorP.forward(state);
Tensor& out = actorP.output();
return noise(out);
```

---

## 优化前后对比

| 项目 | 原始实现 | 优化后 |
|------|---------|--------|
| learningSteps 初始化 | 未初始化 (UB) | `learningSteps(0)` |
| Critic TD-target 位置 | 目标策略最佳动作 | 实际执行动作 |
| Critic Q-target 构建 | 覆盖目标动作 + 保留其他 | 复制 Q 后仅改目标动作 |
| Actor 策略梯度 | hard argmax one-hot 推动 | 正确 Q-value 梯度 `-Q` |
| 目标网络更新 | 每 replaceTargetIter 步 | 每步 Polyak (tau=5e-3) |
| 优化器 | RMSProp (不同参数) | Adam (统一 1e-3) |
| noiseAction | 原地修改网络输出 | 使用 output() 引用 |

## 核心公式

### Critic Loss (TD Learning)
$$L_Q = \mathbb{E}_{(s,a,r,s') \sim D} \left[ (Q(s,a) - y)^2 \right]$$

其中 TD-target:
$$y = \begin{cases}
r & \text{if } s' \text{ is terminal} \\
r + \gamma \cdot Q'(s', \arg\max \pi'(s')) & \text{otherwise}
\end{cases}$$

### Policy Gradient (Deterministic Policy Gradient for discrete actions)
$$J(\pi) = \mathbb{E}_{s \sim D} \left[ \sum_{a} \pi(a|s) \cdot Q(s,a) \right]$$

$$\nabla_{\theta} J = \mathbb{E}_{s \sim D} \left[ \sum_{a} \frac{\partial \pi(a|s)}{\partial \theta} \cdot Q(s,a) \right]$$

经过 Softmax 链式法则，输出层梯度为：
$$\delta_i = \pi_i \cdot (Q_i - \sum_j \pi_j \cdot Q_j)$$

这等价于直接对 Softmax 输出求 `-Q_i` 的梯度，因为 `backward` 通过 Jacobian 矩阵自动完成上述转换。

### Polyak Averaging (目标网络更新)
$$ \theta'_{\text{target}} \leftarrow \tau \cdot \theta_{\text{online}} + (1 - \tau) \cdot \theta_{\text{target}} $$

---

## 算法伪代码

```
Algorithm DDPG (离散动作空间版本):

初始化 Actor π(θ), Critic Q(φ)
初始化 Target Actor π'(θ'), Target Critic Q'(φ')
θ' ← θ, φ' ← φ
初始化经验回放池 D

for each episode:
    s ← 初始状态
    for each step:
        a ← gumbelSoftmax(π(s))  # 探索
        s', r, done ← env.step(a)
        D.push(s, a, s', r, done)
        s ← s'
        
        if len(D) > batch_size:
            for i in 1..batch_size:
                s_i, a_i, s'_i, r_i, done_i ← D.sample()
                
                # 训练 Critic
                Q_current = Q(s_i)
                a'_best = argmax π'(s'_i)
                y_i = r_i + γ(1 - done_i) · Q'(s'_i)[a'_best]
                Q_target = Q_current; Q_target[a_i] = y_i
                ∇_φ ||Q(s_i) - Q_target||²
                
                # 训练 Actor
                ∇_θ J = -Q(s_i)  # backward through π's softmax
                
                # 软更新目标网络
                θ' ← τ·θ + (1-τ)·θ'
                φ' ← τ·φ + (1-τ)·φ'
```

## 修改文件

- **rl/ddpg.cpp**: 重写构造函数（初始化 `learningSteps`）、`experienceReplay()`（修正 Critic 和 Actor 训练逻辑）、`learn()`（每步 Polyak 更新、Adam 优化器）
- **rl/ddpg.h**: 更新注释文档
