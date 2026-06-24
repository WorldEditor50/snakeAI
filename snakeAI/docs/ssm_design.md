# SSM (State Space Model) 设计文档

## 1. 概述

本文档描述了一个简单的**离散时间线性状态空间模型 (SSM)** 的实现。该模型作为序列数据处理的基本单元，遵循与 LSTM 相同的 `iLayer` 接口，并支持通过 BPTT (Backpropagation Through Time) 进行训练。

## 2. 数学原理

### 2.1 核心方程

SSM 将输入序列 `x(t)` 映射到输出 `y(t)`，通过一个隐藏状态 `h(t)` 来捕获时序依赖：

```
h(t) = A · h(t-1) + B · x(t)

y(t) = tanh(C · h(t) + b)
```

其中：
- `h(t) ∈ ℝᵈ` — t 时刻的隐藏状态 (hidden state)，d = hiddenDim
- `x(t) ∈ ℝⁿ` — t 时刻的输入，n = inputDim
- `y(t) ∈ ℝᵐ` — t 时刻的输出，m = outputDim
- `A ∈ ℝᵈˣᵈ` — 状态转移矩阵 (state transition matrix)
- `B ∈ ℝᵈˣⁿ` — 输入投影矩阵 (input projection)
- `C ∈ ℝᵐˣᵈ` — 输出投影矩阵 (output projection)
- `b ∈ ℝᵐ` — 输出偏置 (output bias)

### 2.2 状态更新机制

状态更新由两部分组成：

1. **自演化**: `A · h(t-1)` — 保持前一步的记忆，A 控制信息的衰减/增强
2. **输入驱动**: `B · x(t)` — 将当前输入融入状态

初始化时 A 被设置为**近似单位矩阵**（对角线 0.95~0.99，非对角线小随机值）。这确保：
- 状态能够稳定地随时间传播（特征值接近 1）
- 梯度在长序列中不会爆炸/消失

### 2.3 输出机制

隐藏状态通过 `C` 投影到输出空间，再经过 `tanh` 激活，产生范围在 [-1, 1] 的输出。

## 3. 架构设计

### 3.1 类层次

```
iLayer (抽象基类)
  └── SSM (状态空间模型)
```

SSM 实现了 `iLayer` 的全部纯虚方法，包括：

| 方法 | 功能 |
|------|------|
| `forward(x, inference)` | 单步前向传播 |
| `cacheError(e)` | 缓存每个时间步的误差 |
| `RMSProp(lr, rho, decay, clipGrad)` | RMSProp 优化（触发 BPTT） |
| `Adam(lr, ...)` | Adam 优化（触发 BPTT） |
| `SGD(lr)` | SGD 优化（触发 BPTT） |
| `copyTo(layer)` | 硬参数复制 |
| `softUpdateTo(layer, alpha)` | 软参数更新 (Polyak averaging) |
| `write/read(file)` | 序列化保存/加载 |
| `clamp(c0, cn)` | 参数裁剪 |
| `reset()` | 重置状态和缓存 |

### 3.2 状态管理

**SSM::State** 结构体记录每个时间步的完整状态：

```cpp
class State {
    Tensor h;   // 隐藏状态 (hiddenDim × 1)
    Tensor y;   // 输出     (outputDim × 1)
};
```

前向传播时，每个时间步的 `State` 被缓存到 `states` 向量中。BPTT 反向遍历这些状态计算梯度。

### 3.3 与 Net 的集成

在 `Net::backward()` 中，当检测到 `LAYER_SSM` 类型的层时，会采用与 LSTM 相同的处理策略：

```cpp
} else if(layers[i - 1]->type == iLayer::LAYER_SSM) {
    Tensor e(layers[i - 1]->o.totalSize, 1);
    layers[i]->backward(e);
    layers[i - 1]->cacheError(e);  // BPTT via cacheError
}
```

优化器调用时（如 `RMSProp`），SSM 内部会：
1. 运行 `backward(cacheX, cacheE)` 执行完整 BPTT
2. 应用梯度更新参数
3. 清零梯度缓存

## 4. BPTT 梯度推导

### 4.1 前向传播 (一个时间步)

```
z(t) = C · h(t) + b
y(t) = tanh(z(t))
```

### 4.2 梯度计算

给定输出误差 `E = ∂L/∂y(t)`，参数梯度为：

#### 通过 tanh 传播
```
δy(t) = E ⊙ tanh'(y(t))
```
其中 `tanh'(y) = 1 - y²`

#### 输出参数梯度
```
∂L/∂C += δy(t) · h(t)ᵀ    (ikjk 外积)
∂L/∂b += δy(t)             (向量加)
```

#### 隐藏状态梯度
```
δh(t)_C = Cᵀ · δy(t)         (来自输出路径)
δh(t)_A = Aᵀ · δh(t+1)       (来自未来时间步的反向传播)
δh(t) = δh(t)_C + δh(t)_A    (总梯度)
```

#### 状态参数梯度
```
∂L/∂A += δh(t) · h(t-1)ᵀ    (外积)
∂L/∂B += δh(t) · x(t)ᵀ      (外积)
```

#### 梯度传播到前一个时间步
```
δh(t-1) = Aᵀ · δh(t)
```

### 4.3 BPTT 算法流程

```
backward(cacheX, cacheE):
    初始化 δh = 0 (最后一个时间步之后没有未来梯度)
    for t = T-1 ... 0 (反向遍历):
        backwardAtTime(t, cacheX[t], cacheE[t], δh)
    states.clear()  // 清理缓存

backwardAtTime(t, x, E, δh_next):
    δy = E ⊙ tanh'(y[t])
    δh = Cᵀ · δy
    δh += Aᵀ · δh_next          // 加上来自未来的梯度
    
    g.C += δy · h[t]ᵀ
    g.b += δy
    g.A += δh · h[t>0 ? t-1 : 0]ᵀ   // t=0 时使用初始零状态
    g.B += δh · xᵀ
    
    if t > 0:
        δh_next = Aᵀ · δh       // 传播到 t-1
```

## 5. 参数初始化

### A 矩阵（状态转移）
```
对角线元素: A[i,i] ~ Uniform(0.95, 0.99)  — 接近 1 保证稳定性
非对角线:   A[i,j] ~ Uniform(-0.02, 0.02) — 小随机扰动
```

### B, C 矩阵
```
B[i], C[i] ~ Uniform(-0.1, 0.1)  — 小值初始化
```

### b 偏置
```
b[i] = 0  — 零初始化
```

## 6. 测试方案

### 6.1 单元测试列表

| 测试 | 验证内容 |
|------|---------|
| Test 1 | 构造器：维度正确、类型注册 |
| Test 2 | 单步前向：tanh 输出范围 [-1, 1] |
| Test 3 | 时序状态演化：连续两步状态不同 |
| Test 4 | BPTT 梯度流：A/B/C 梯度非零 |
| Test 5 | 训练：MSE 损失在 `sin(x²+y²)` 任务上下降 |
| Test 6 | Reset：清除所有缓存和隐藏状态 |
| Test 7 | copyTo/softUpdateTo：参数正确复制/混合 |
| Test 8 | Save/Load：序列化完整保持所有参数 |
| Test 9 | 推理模式：不缓存任何数据 |

### 6.2 训练性能

在 200 个样本的 `sin(x²+y²)` 回归任务上，使用 hiddenDim=10 的 SSM：
- 初始 MSE: ~1.58
- 经过 200 次 RMSProp 迭代后 MSE: ~0.15
- 损失成功下降表明 BPTT 梯度正确且优化器有效

## 7. 与 LSTM 的对比

| 特性 | SSM | LSTM |
|------|-----|------|
| 参数量 | O(d² + d·n + m·d) | O(4·d² + 4·d·n + m·d) |
| 状态维数 | 1 (h) | 2 (h, c) |
| 门控机制 | 无 | 输入门/遗忘门/输出门 |
| 非线性 | tanh 输出 | tanh + sigmoid 门控 |
| 长程依赖 | 依赖 A 初始化 | 内置遗忘门控制 |
| 计算效率 | 更高（矩阵乘法少） | 较低（4倍参数） |

SSM 作为更轻量的序列模型，适合计算资源受限或需要快速推理的场景。其性能高度依赖于 A 矩阵的初始化——接近单位矩阵的对角线初始化是实现稳定长程记忆的关键。
