# LSTM 梯度计算合理性评估报告

## 评估范围

对 `rl/lstm.cpp` 中的前向传播 (`feedForward`) 和反向传播 (`backwardAtTime`) 进行数学推导验证。

---

## 一、前向传播公式验证

### LSTM基本架构

```
f_t = σ(Wf·x_t + Uf·h_{t-1} + bf)   遗忘门
i_t = σ(Wi·x_t + Ui·h_{t-1} + bi)   输入门
g_t = tanh(Wg·x_t + Ug·h_{t-1} + bg) 候选细胞
o_t = σ(Wo·x_t + Uo·h_{t-1} + bo)   输出门
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      细胞状态
h_t = o_t ⊙ tanh(c_t)                隐藏状态
y_t = tanh(W·h_t + b)                输出层
```

### 代码实现对照

**前向阶段1 — 门控输入计算 (lines 84-99)**
- 正确计算 `W*·x` 和 `U*·h_{t-1}` 并累加

**前向阶段2 — 激活与状态更新 (lines 100-107)**
- `f_t = Sigmoid::f(f_linear + bf)` ✓
- `i_t = Sigmoid::f(i_linear + bi)` ✓
- `g_t = Tanh::f(g_linear + bg)` ✓
- `o_t = Sigmoid::f(o_linear + bo)` ✓
- `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` ✓
- `h_t = o_t ⊙ tanh(c_t)` ✓

**前向阶段3 — 输出层 (lines 109-114)**
- `y_t = Tanh::f(W·h_t + b)` ✓

---

## 二、反向传播公式推导与代码对照

### 符号约定

| 符号 | 含义 |
|------|------|
| E = ∂L/∂y_t | 损失对输出的梯度（由上游传入） |
| δz_y = ∂L/∂z_y | 损失对输出层预激活 z_y = W·h_t + b 的梯度 |
| δh_t | 损失对隐藏状态 h_t 的梯度 |
| δc_t | 损失对细胞状态 c_t 的梯度 |
| δo_t / δi_t / δf_t / δg_t | 损失对各门控预激活的梯度 |

---

### 2.1 输出层梯度

#### 数学推导

输出层：`y_t = tanh(z_y)`, where `z_y = W·h_t + b`

```
δz_y[i] = ∂L/∂z_y[i] = E[i] · tanh'(y_t[i])
∂L/∂W[i,j] = δz_y[i] · h_t[j]
∂L/∂b[i] = δz_y[i]
∂L/∂h_t[j] = Σ_i δz_y[i] · W[i,j] = (W^T · δz_y)[j]
```

#### 代码实现

```cpp
// line 163: 正确计算 δz_y
float error = E[i] * Tanh::df(states[t].y[i]);

// lines 164-166: ∂L/∂W[i,j] = error · h_t[j]  ✓
g.w(i, j) += error * states[t].h[j];

// line 167: ∂L/∂b[i] = error  ✓
g.b[i] += error;

// line 137: ∂L/∂h_t[j] = (W^T · error)  ❌ **注意**
Tensor::MM::kijk(delta.h, w, E);
```

### ⚠️ **发现 Bug 1：输出层 tanh 激活导数缺失** (Medium)

**问题位置**：`lstm.cpp` 第 137 行

**代码**：
```cpp
Tensor::MM::kijk(delta.h, w, E);   // 使用原始 E
```

**正确应该为**：
```cpp
// 应使用经过 tanh 导数调整的 δz_y，而非原始 E
// 即 kijk(delta.h, w, δz_y) 
// 其中 δz_y[i] = E[i] * Tanh::df(states[t].y[i])
```

**影响分析**：
- `delta.h` 中来自输出层的分量缺少了 `tanh'` 导数因子
- 具体缺失为：`W^T · E` 替代了 `W^T · (E ⊙ tanh'(y))`
- 如果 `y_t ∈ (-1,1)` 且 `tanh'(y) ∈ (0,1]`，则梯度被系统性地低估
- 这将影响所有四个门控的梯度以及后续时间步的反向传播

---

### 2.2 隐藏状态到细胞状态的梯度 (δh_t → δc_t)

#### 数学推导

```
h_t = o_t ⊙ tanh(c_t)
∂h_t/∂c_t = o_t ⊙ tanh'(c_t)
∂L/∂c_t (来自h_t) = δh_t ⊙ o_t ⊙ tanh'(c_t)
∂L/∂c_t (来自c_{t+1}) = δc_{t+1} ⊙ f_{t+1}  (通过时间反向传播)
```

#### 代码实现

```cpp
// line 156: 正确实现 ✓
delta.c[i] = delta.h[i] * states[t].o[i] * Tanh::df(states[t].c[i])
           + delta_.c[i] * f_[i];
```

**结论：正确** ✓

---

### 2.3 输出门梯度 (δh_t → δo_t)

#### 数学推导

```
∂L/∂o_t[i] = δh_t[i] · tanh(c_t[i])
∂L/∂z_o[i] = ∂L/∂o_t[i] · σ'(z_o[i]) = δh_t[i] · tanh(c_t[i]) · σ'(o_t[i])
```

#### 代码实现

```cpp
// line 157: 正确实现 ✓
delta.o[i] = delta.h[i] * Tanh::f(states[t].c[i]) * Sigmoid::df(states[t].o[i]);
```

其中 `Sigmoid::df(y) = 1.702·y·(1-y) = σ'(z)`，即 sigmoid 对输入 z 的导数。

**结论：正确** ✓

---

### 2.4 细胞门梯度 (δc_t → δg_t)

#### 数学推导

```
∂L/∂g_t[i] = δc_t[i] · i_t[i]
∂L/∂z_g[i] = δc_t[i] · i_t[i] · tanh'(g_t[i])
```

#### 代码实现

```cpp
// line 158: 正确实现 ✓
delta.g[i] = delta.c[i] * states[t].i[i] * Tanh::df(states[t].g[i]);
```

**结论：正确** ✓

---

### 2.5 输入门梯度 (δc_t → δi_t)

#### 数学推导

```
∂L/∂i_t[i] = δc_t[i] · g_t[i]
∂L/∂z_i[i] = δc_t[i] · g_t[i] · σ'(i_t[i])
```

#### 代码实现

```cpp
// line 159: 正确实现 ✓
delta.i[i] = delta.c[i] * states[t].g[i] * Sigmoid::df(states[t].i[i]);
```

**结论：正确** ✓

---

### 2.6 遗忘门梯度 (δc_t → δf_t)

#### 数学推导

```
∂L/∂f_t[i] = δc_t[i] · c_{t-1}[i]
∂L/∂z_f[i] = δc_t[i] · c_{t-1}[i] · σ'(f_t[i])
```

#### 代码实现

```cpp
// line 160: 正确实现 ✓
delta.f[i] = delta.c[i] * _c[i] * Sigmoid::df(states[t].f[i]);
```

其中 `_c = t > 0 ? states[t-1].c : zeros`。

**结论：正确** ✓

---

### 2.7 参数梯度累加

#### 权重矩阵梯度

```
∂L/∂W* = δz_* · x_t^T
∂L/∂U* = δz_* · h_{t-1}^T
```

**代码实现**：
```cpp
// lines 170-173: ✓
Tensor::MM::ikjk(g.wi, delta.i, x);    // g.wi += δi · x^T
Tensor::MM::ikjk(g.wf, delta.f, x);    // g.wf += δf · x^T
Tensor::MM::ikjk(g.wg, delta.g, x);    // g.wg += δg · x^T
Tensor::MM::ikjk(g.wo, delta.o, x);    // g.wo += δo · x^T

// lines 175-178: ✓
Tensor::MM::ikjk(g.ui, delta.i, _h);   // g.ui += δi · h_{t-1}^T
Tensor::MM::ikjk(g.uf, delta.f, _h);   // g.uf += δf · h_{t-1}^T
Tensor::MM::ikjk(g.ug, delta.g, _h);   // g.ug += δg · h_{t-1}^T
Tensor::MM::ikjk(g.uo, delta.o, _h);   // g.uo += δo · h_{t-1}^T
```

#### 偏置梯度

```
∂L/∂b* = δz_*
```

```cpp
// lines 180-183: ✓
g.bi += delta.i;
g.bf += delta.f;
g.bg += delta.g;
g.bo += delta.o;
```

**结论：参数梯度累加正确** ✓

---

### 2.8 时间反向传播

```cpp
// line 193: 从最后一个时间步反向遍历 ✓
for (int t = states.size() - 1; t >= 0; t--) {
    backwardAtTime(t, x[t], E[t], delta_);
}
```

`delta_` 被传递给每个时间步，其中包含来自 `t+1` 时间步的细胞状态梯度 (`delta_.c`) 和各门控梯度 (`delta_.i/f/g/o`)。

```cpp
// line 185: 更新 delta_ 以传递给下一个（更早的）时间步
delta_ = delta;
```

**结论：BPTT（沿时间反向传播）机制正确** ✓

---

## 三、边界条件分析

### 3.1 第一个时间步 (t=0)

```cpp
// line 153: c_{-1} = 0
Tensor _c = t > 0 ? states[t - 1].c : Tensor(hiddenDim, 1);

// line 174: h_{-1} = 0
Tensor _h = t > 0 ? states[t - 1].h : Tensor(hiddenDim, 1);
```

- `delta.f[i] = ... * 0 ... = 0` — 合理，因为没有历史细胞状态可遗忘 ✓
- `g.U* += δ* · 0 = 0` — 合理，第一个时间步不产生U梯度 ✓
- `g.w* += δ* · x` — 仍有输入`x`的梯度 ✓

### 3.2 最后一个时间步 (t=T-1)

```cpp
// line 152: f_{T} = 0（无未来遗忘门）
Tensor f_ = t < states.size() - 1 ? states[t + 1].f : Tensor(hiddenDim, 1);
```

- `delta.c[i]` 只包含来自 `δh_t` 的分量，无来自 `c_{t+1}` 的分量 ✓
- `delta_.i/f/g/o` 初始为零，不贡献 `kijk` 累加 ✓

### 3.3 空状态重置

```cpp
// lines 191-197: 反向传播后清除 states
void RL::LSTM::backward(...) {
    ...
    states.clear();
}
```

`reset()` 也清零 `h`、`c`、`cacheX`、`cacheE`、`states`。

**结论：边界条件处理正确** ✓

---

## 四、与前向传播的一致性验证

| 前向公式 | 反向梯度 | 代码 | 状态 |
|---------|---------|------|------|
| y = tanh(W·h+b) | δW = (E⊙tanh'(y))·h^T | line 164 | ✅ |
| y = tanh(W·h+b) | δh = W^T·(E⊙tanh'(y)) | line 137 | ❌ 缺少tanh' |
| h = o⊙tanh(c) | δc += δh⊙o⊙tanh'(c) | line 156 | ✅ |
| h = o⊙tanh(c) | δo = δh⊙tanh(c)⊙σ'(o) | line 157 | ✅ |
| c += i⊙g | δg = δc⊙i⊙tanh'(g) | line 158 | ✅ |
| c += i⊙g | δi = δc⊙g⊙σ'(i) | line 159 | ✅ |
| c = f⊙c' + i⊙g | δf = δc⊙c'⊙σ'(f) | line 160 | ✅ |

---

## 五、发现的 Bug 汇总

### Bug 1 (Medium): 输出层 tanh 导数缺失

**文件**: `rl/lstm.cpp`  
**行号**: 137  
**严重程度**: 中等  

**问题描述**：
```cpp
// 当前代码 (行137): 使用了原始 E
Tensor::MM::kijk(delta.h, w, E);

// 应改为: 使用经过 tanh 导数调整的 error
// 需要先构造一个包含 error 的 Tensor，或用其他方式传递
```

**影响**：梯度从输出层反向传播到 LSTM 隐藏状态时缺失了 `tanh` 激活函数的导数因子，导致所有门控的梯度都被系统性地低估。由于 BPTT 会进一步前传这个偏差，较早时间步的受影响程度逐步累积。

**修复方案**：
```cpp
// 在行162之前构造 error 张量
Tensor outputError(hiddenDim, 1);
for (std::size_t i = 0; i < outputError.size(); i++) {
    outputError[i] = E[i] * Tanh::df(states[t].y[i]);
}
// 行137改为:
Tensor::MM::kijk(delta.h, w, outputError);
```

**或更简洁的修复**（复用已有计算）：
将原本的行137移动或重构，使 tanh 导数的计算在 `kijk` 调用之前完成。

### 与文档中已记录 Bug 的关联

| 文档中的 Bug | 当前代码状态 | 与 LSTM 关系 |
|-------------|-------------|-------------|
| kijk 自赋值 (tensor.hpp L970) | ✅ 已修复 | 不影响当前 LSTM 代码 |
| 参数错位 (concat/attention) | 未修复 | 影响使用 concat/attention 的 Net，不直接影响 LSTM |
| PPO TD 目标 | 未修复 | 不影响 LSTM |

---

## 六、总体评估

| 评估维度 | 结果 | 说明 |
|---------|------|------|
| 前向传播正确性 | ✅ **正确** | LSTM 标准公式完整实现 |
| 反向传播公式推导 | ✅ **正确** | 所有门控梯度公式与标准 BPTT 一致 |
| 参数梯度累加 | ✅ **正确** | 权重、偏置梯度计算正确 |
| 边界条件处理 | ✅ **正确** | t=0, t=T-1 均处理得当 |
| **输出层梯度传递** | ❌ **存在 Bug** | 缺少 tanh 导数因子 (Bug 1) |
| 数值稳定性 | ⚠️ 可改进 | Sigmoid 近似 (1.702x) 与标准 sigmoid 略有偏差 |
| 梯度裁剪 | ✅ | 优化器中已实现 |

### 最终结论

LSTM 的梯度计算**整体框架正确**，遵循标准的 BPTT（Backpropagation Through Time）算法。四个门控的局部梯度推导和实现均无误，时间维度的梯度传播机制正确，边界条件处理得当。

**存在一个中等严重程度的 Bug**：输出层的 `tanh` 激活函数导数在反向传播到 LSTM 隐藏状态时被遗漏（第137行），这导致从输出层到 LSTM 核心的梯度通路系统性偏小。

建议在实际训练中：
1. **优先修复 Bug 1**，以确保梯度精确性
2. 如果输出层改为 Linear 激活（即 `y = W·h + b`），则此 Bug 自动消除（因为 `linear'(x)=1`）
3. 对于梯度噪声容忍度较高的 RL 任务，此 Bug 可能不会完全阻碍收敛，但会影响训练效率
