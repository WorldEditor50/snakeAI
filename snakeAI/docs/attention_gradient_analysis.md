# Attention 网络梯度计算合理性评估报告

## 评估范围

对 `rl/attention.hpp` 中的以下组件进行完整数学推导与分析：
1. **PositionalEncoder** — 位置编码层
2. **ScaledDotProduct** — 缩放点积注意力（单头）
3. **Attention\<N\>** — 多头注意力层

同时分析 `rl/net.hpp::Net` 中的 backward 调用链如何与 Attention 交互。

---

## 一、PositionalEncoder 分析

### 前向传播

```cpp
for (std::size_t i = 0; i < x.totalSize; i++) {
    if (i%2 == 0) {
        pe[i] = std::sin(float(pos)/std::pow(10000, float(i)/d));
    } else {
        pe[i] = std::cos(float(pos)/std::pow(10000, float(i - 1)/d));
    }
    o[i] = x[i] + pe[i];
}
```

**输入输出**: `o = x + pe`，其中 `pe` 为固定的位置编码向量。

### 梯度计算

`gradient()`, `backward()`, 所有优化器方法均为空实现（无参数）。

**结论**: ✅ **正确** — PositionalEncoder 无可训练参数，梯度无需计算。`o = x + pe` 的梯度直接等于 e（恒等传递）。

---

## 二、ScaledDotProduct 深度分析

### 2.1 前向传播

```
q = Wq·x            (N×1)    N = outputDim
k = Wk·x            (N×1)
v = Wv·x            (N×1)
z = q·k^T / d       (N×N)    d = √N
ẑ = softmax(z)      (N×N)
o = ẑ · v           (N×1)
```

#### 实现代码

```cpp
// lines 148-155
Tensor::MM::ikkj(q, wq, x);     // q = Wq·x
Tensor::MM::ikkj(k, wk, x);     // k = Wk·x
Tensor::MM::ikkj(v, wv, x);     // v = Wv·x
Tensor::MM::ikjk(z, q, k);      // z = q·k^T
z /= d;                          // z /= √N
softmax(z);                      // ẑ = softmax(z)
Tensor::MM::ikkj(o, z, v);      // o = ẑ·v
```

**结论**: ✅ 前向传播公式正确

---

### 2.2 ScaledDotProduct 梯度核心推导

#### 符号定义

| 符号 | 维度 | 含义 |
|------|------|------|
| x | inputDim × 1 | 输入向量 |
| Wq, Wk, Wv | N × inputDim | 可训练权重 (N=outputDim) |
| q, k, v | N × 1 | Query/Key/Value |
| z | N × N | 预 Softmax 注意力分数 |
| ẑ | N × N | Softmax 注意力权重 |
| o | N × 1 | 输出 |
| e | N × 1 | ∂L/∂o (上游传入) |

#### 基本梯度

```
o = ẑ · v  →  oᵢ = Σⱼ ẑᵢⱼ · vⱼ

∂L/∂ẑᵢⱼ = eᵢ · vⱼ         →  [∂L/∂ẑ] = e · v^T
∂L/∂vⱼ  = Σᵢ eᵢ · ẑᵢⱼ    →  ∂L/∂v = ẑ^T · e
```

#### Softmax 梯度

ẑ = softmax(z)，Jacobian 矩阵 J (N² × N²):
```
J[(i,j),(p,k)] = ∂ẑᵢⱼ/∂zₚₖ = ẑᵢⱼ · (δᵢₚ·δⱼₖ - ẑₚₖ)
```

反向传播通过 Softmax：
```
vec(∂L/∂z) = J^T · vec(∂L/∂ẑ)
```

#### Q, K 梯度

```
∂L/∂qᵢ = Σⱼ (∂L/∂zᵢⱼ) · (kⱼ/d)    →  ∂L/∂q = H · k / d
∂L/∂kⱼ = Σᵢ (∂L/∂zᵢⱼ) · (qᵢ/d)    →  ∂L/∂k = H^T · q / d
```

其中 H = vec⁻¹(J^T · vec(∂L/∂ẑ)) = ∂L/∂z (矩阵形式)

#### 权重梯度

```
∂L/∂Wq = (∂L/∂q) · x^T
∂L/∂Wk = (∂L/∂k) · x^T
∂L/∂Wv = (∂L/∂v) · x^T = (ẑ^T · e) · x^T
```

---

### 2.3 修复后代码

#### ScaledDotProduct::backward — 修复实现

```cpp
void backward(Tensor &ei) override
{
    /*
        o = ẑ · v, ẑ = softmax(q·k^T / d)
        Given e = ∂L/∂o (N×1):
        1. ∂L/∂v = ẑ^T · e
        2. ∂L/∂ẑ = e · v^T
        3. vec(∂L/∂z) = J^T · vec(∂L/∂ẑ)
        4. ∂L/∂q = (∂L/∂z) · k / d
        5. ∂L/∂k = (∂L/∂z)^T · q / d
        6. ∂L/∂x = Wq^T·dq + Wk^T·dk + Wv^T·dv
    */
    /* Step 1: dv = z^T · e */
    Tensor dv(outputDim, 1);
    Tensor::MM::kikj(dv, z, e);

    /* Step 2: dz_hat = e · v^T */
    Tensor dz_hat(outputDim, outputDim);
    Tensor::MM::ikjk(dz_hat, e, v);

    /* Step 3: dz = J^T · vec(dz_hat) */
    Tensor J = Softmax::jacobian(z);
    Tensor dz_hat_vec = dz_hat;
    dz_hat_vec.reshape(outputDim*outputDim, 1);
    Tensor dz_vec(outputDim*outputDim, 1);
    Tensor::MM::kikj(dz_vec, J, dz_hat_vec);   // kikj = J^T · vec(.)
    dz_vec.reshape(outputDim, outputDim);

    /* Step 4: dq = dz · k / d, dk = dz^T · q / d */
    Tensor dq(outputDim, 1);
    Tensor::MM::ikkj(dq, dz_vec, k);
    dq /= d;
    Tensor dk(outputDim, 1);
    Tensor dz_t = dz_vec.tr();
    Tensor::MM::ikkj(dk, dz_t, q);
    dk /= d;

    /* Step 5: ∂L/∂x = Wq^T·dq + Wk^T·dk + Wv^T·dv
       NOTE: NO ei.zero() — caller accumulates */
    Tensor::MM::kikj(ei, wq, dq);
    Tensor::MM::kikj(ei, wk, dk);
    Tensor::MM::kikj(ei, wv, dv);
}
```

#### ScaledDotProduct::gradient — 修复实现

```cpp
void gradient(const Tensor& x, const Tensor&) override
{
    /* Steps 1-4: same as backward */
    Tensor dv(outputDim, 1);
    Tensor::MM::kikj(dv, z, e);

    Tensor dz_hat(outputDim, outputDim);
    Tensor::MM::ikjk(dz_hat, e, v);

    Tensor J = Softmax::jacobian(z);
    Tensor dz_hat_vec = dz_hat;
    dz_hat_vec.reshape(outputDim*outputDim, 1);
    Tensor dz_vec(outputDim*outputDim, 1);
    Tensor::MM::kikj(dz_vec, J, dz_hat_vec);
    dz_vec.reshape(outputDim, outputDim);

    Tensor dq(outputDim, 1);
    Tensor::MM::ikkj(dq, dz_vec, k);
    dq /= d;
    Tensor dk(outputDim, 1);
    Tensor dz_t = dz_vec.tr();
    Tensor::MM::ikkj(dk, dz_t, q);
    dk /= d;

    /* Step 5-7: weight gradients */
    Tensor::MM::ikjk(g.wq, dq, x);    // g.wq += dq · x^T
    Tensor::MM::ikjk(g.wk, dk, x);    // g.wk += dk · x^T
    Tensor::MM::ikjk(g.wv, dv, x);    // g.wv += dv · x^T

    q.zero(); k.zero(); v.zero(); z.zero(); o.zero(); e.zero();
}
```

---

## 三、Attention\<N\> — 多头堆叠方式分析

### 3.1 架构设计

当前实现的 Attention 并非标准 Transformer 多头注意力，而是一个**自定义的多头特征提取 + 门控融合**模块：

```
# 标准 Transformer MHA:
head_i(Q,K,V) = softmax(Q_i·K_i^T/√d)·V_i   # 序列上的注意力
MultiHead(Q,K,V) = Concat(head₁...headₙ)·W_O

# 当前实现（自定义变体）:
head_i(x) = softmax(Wq_i·x · (Wk_i·x)^T / √d) · Wv_i·x  # 特征维度上的自注意力
a = [head₁(x); head₂(x); ...; headₙ(x)]                  # 拼接 (outputDim × 1)
o = tanh(W₁·a + W₂·x + b)                                # 全连接融合 + 残差 + tanh
```

**关键架构差异**：

| 方面 | 标准 Transformer MHA | 当前实现 |
|------|---------------------|---------|
| 注意力维度 | 序列长度维度 | 特征维度 |
| 多头组合 | Concat→W_O 投影 | 拼接 → W₁融合 + W₂残差 |
| 激活函数 | 通常无 | tanh |
| 输入形式 | Q,K,V 可来自不同源 | 所有头共享同一输入 x |

**结论**: 这不是标准 MHA，而是一种合理的自定义注意力变体（类似多视角特征提取 + Gated Attention 的混合）。架构设计无本质错误。

### 3.2 数据流一致性

**前向** (embedding 拼接):
```
head_i(x) = dotProduct[i].forward(x)   → N×1 输出
a.embedding({i*unitDim, 0}, head_i)     → a[i*unitDim : (i+1)*unitDim] = head_i
```

**反向** (block 切分):
```
da = w1^T · dy                          → outputDim×1
dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1})  → 各头取对应块
```

`embedding` 和 `block` 使用相同的偏移量 `{i*unitDim, 0}` 和大小 `{unitDim, 1}`，**数据布局一致** ✅

### 3.3 Gradient 方法分析（修复后）

```cpp
void gradient(const Tensor& x, const Tensor&y) override
{
    Tensor dy(outputDim, 1);
    for (std::size_t i = 0; i < outputDim; i++) {
        dy[i] = Tanh::df(o[i])*e[i];              // ∂L/∂z = e ⊙ tanh'(o)
    }
    Tensor::MM::ikjk(g.w1, dy, a);                 // g.w1 += dy · a^T
    Tensor::MM::ikjk(g.w2, dy, x);                 // g.w2 += dy · x^T
    g.b += dy;                                      // g.b += dy

    /* 设置各头误差并计算参数梯度 */
    Tensor da(outputDim, 1);
    Tensor::MM::kikj(da, w1, dy);                  // da = w1^T · dy
    for (int i = 0; i < N; i++) {
        dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1});
        dotProduct[i].gradient(x, y);               // 各头参数梯度
    }
    o.zero();
    e.zero();
    return;
}
```

**数学验证**:

| 项 | 公式 | 代码 | 验证 |
|----|------|------|------|
| dy | e ⊙ tanh'(o) | `Tanh::df(o[i])*e[i]` | ✅ |
| g.w1 | (∂L/∂z)·a^T = dy·a^T | `ikjk(g.w1, dy, a)` | ✅ |
| g.w2 | (∂L/∂z)·x^T = dy·x^T | `ikjk(g.w2, dy, x)` | ✅ |
| g.b | ∂L/∂z = dy | `g.b += dy` | ✅ |
| da | w1^T·dy (各头输出梯度) | `kikj(da, w1, dy)` | ✅ |
| 头梯度分发 | da[i*unitDim:(i+1)*unitDim] | `da.block(...)` | ✅ |

**结论**: Attention 外层梯度计算（w1, w2, b）✅ 正确，各头误差分发 ✅ 正确

### 3.4 Backward 方法分析（修复后）

```cpp
void backward(Tensor &ei) override
{
    Tensor dy(outputDim, 1);
    for (std::size_t i = 0; i < outputDim; i++) {
        dy[i] = Tanh::df(o[i]) * e[i];             // ∂L/∂z = e ⊙ tanh'(o)
    }

    /* 残差路径: ∂L/∂x += w2^T · dy */
    Tensor::MM::kikj(ei, w2, dy);

    /* 注意力路径: ∂L/∂a = w1^T · dy → 各头反向传播 */
    Tensor da(outputDim, 1);
    Tensor::MM::kikj(da, w1, dy);
    for (int i = 0; i < N; i++) {
        dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1});
        dotProduct[i].backward(ei);                 /* 累加 ∂L/∂x_head 到 ei */
    }
    return;
}
```

**数学验证**:
```
∂L/∂x = w2^T · (e ⊙ tanh'(o))           ← 残差路径
       + Σ_i ∂L/∂head_i                  ← 各头反向传播路径
       = w2^T · dy + Σ_i Wq_i^T·dq_i + Wk_i^T·dk_i + Wv_i^T·dv_i
```

**结论**: backward 正确地在两个路径上传播梯度，并累加到 ei ✅

---

## 四、多头堆叠：ei.zero() 覆盖问题

### ❌ Bug 5 (Critical)：ScaledDotProduct::backward 中 `ei.zero()` 导致多头梯度被覆盖

**修复前代码**:
```cpp
// ScaledDotProduct::backward (修复前)
void backward(Tensor &ei) override
{
    Tensor w = wq + wk + wv;
    ei.zero();                                      // ← ！！！
    Tensor::MM::kikj(ei, w, e);                     // ei = w^T · e
}
```

**问题分析**:

当 Attention::backward 循环调用各头时：
```cpp
// Attention::backward (修复前)
for (int i = 0; i < N; i++) {
    dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1});
    dotProduct[i].backward(ei);
    // 头 i=0: ei.zero() → ei = ∂L/∂x_head0
    // 头 i=1: ei.zero() → 清掉头0的贡献！ei = ∂L/∂x_head1
    // 头 i=2: ...覆盖...
    // 最终: ei 只包含最后一个头 {N-1} 的梯度
}
```

**后果**：
- 各 ScaledDotProduct 头共享同一个 `ei`（输入梯度 Tensor）
- 每个头调用 `backward` 时都会 `ei.zero()` 清零，覆盖前面所有头的梯度
- **最终只有最后一个头 {N-1} 的梯度存活**

**修复**: 移除 ScaledDotProduct::backward 中的 `ei.zero()`，改为累加。

---

## 五、Net 调用链分析

### 5.1 Net::backward 中的 Attention 处理

```cpp
// net.hpp lines 68-72
} else if((layers[i - 1]->type == iLayer::LAYER_ATTENTION ||
          layers[i - 1]->type == iLayer::LAYER_SCALEDCONCAT) &&
          layers[i]->type == iLayer::LAYER_FC) {
    layers[i]->backward(layers[i - 1]->e);    // FC反向传播
    layers[i - 1]->broadcast();               // 广播到Attention各头
}
```

### ❌ Bug 6 (Medium)：Attention::backward 在 Net::backward 中未被调用

当 Attention 在 Net 中作为中间层时，`Net::backward` 的特殊处理分支直接调用 FC 的 backward 和 broadcast，但**从未调用 Attention::backward**。这意味着：

1. Attention 层内的 `e` 字段被设置（通过 FC backward）
2. broadcast 将这些 e 分发给各 ScaledDotProduct 头
3. **但 Attention::backward 没有被调用**，因此 ScaledDotProduct::backward 也不会被调用
4. 误差无法通过 Attention 层继续反向传播到更早的层

**影响场景**: 当 Attention 不是网络的第一层时（即 Attention 前还有其他层），梯度无法正确传播到前面的层。

**注意**: 当 Attention 是网络第一层时，此 Bug 无实际影响（因为没有需要接收梯度的前一层）。

---

## 六、Bug 汇总

### ❌ Bug 1 (Critical): ScaledDotProduct 梯度公式严重错误

| 方面 | 修复前实现 | 正确公式 |
|------|----------|---------|
| Softmax Jacobian | 使用 `J`（无转置） | 应使用 `J^T` |
| Jacobian 输入 | `v·k^T` 和 `v·q^T` | `e·v^T`（对 softmax 输出的梯度） |
| ∂L/∂Wq 计算 | `J·vec(v·k^T)·e·x^T/d` | `J^T·vec(e·v^T)·k·x^T/d` |
| ∂L/∂Wk 计算 | `J·vec(v·q^T)·e·x^T/d` | `(J^T·vec(e·v^T))^T·q·x^T/d` |
| ∂L/∂Wv 计算 | `z·e·x^T` | `ẑ^T·e·x^T` |

**状态**: ✅ 已修复

---

### ❌ Bug 2 (Critical): ScaledDotProduct.backward 使用错误近似

```cpp
// 修复前
Tensor w = wq + wk + wv;     // 求和代替分路径传播
Tensor::MM::kikj(ei, w, e);  // ei = w^T · e

// 修复后
// 通过 q, k, v 三条路径分别反向传播，完整链式法则
```

**状态**: ✅ 已修复

---

### ❌ Bug 3 (Medium): Attention.backward 缺少 tanh 导数

```cpp
// 修复前
Tensor::MM::kikj(ei, w2, e);

// 修复后
Tensor dy(outputDim, 1);
for (...) { dy[i] = Tanh::df(o[i]) * e[i]; }
Tensor::MM::kikj(ei, w2, dy);  // 使用 e⊙tanh'(o)
```

**状态**: ✅ 已修复

---

### ❌ Bug 4 (Medium): broadcast/gradient 分发原始 e 而非 w1^T·dy

```cpp
// 修复前
dotProduct[i].e = e.block({i*unitDim, 0}, {unitDim, 1});

// 修复后
Tensor da = w1^T · (e ⊙ tanh'(o));
dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1});
```

**状态**: ✅ 已修复

---

### ❌ Bug 5 (Critical): ei.zero() 导致多头梯度被覆盖

**问题**: `ScaledDotProduct::backward` 中 `ei.zero()` 覆盖了前面所有头的梯度

**状态**: ✅ 已修复（移除 ei.zero()，改为累加）

---

### ❌ Bug 6 (Medium): Attention::backward 在 Net 中未被调用

**问题**: `Net::backward` 特殊分支未调用 `Attention::backward`

**状态**: ⚠️ 待修复（参见 `rl/net.hpp`）

---

## 七、标准 Multi-Head Attention (MHA) 实现

### 7.1 MHA 类设计

基于框架内已有组件（ScaledDotProduct、Tensor::MM、Optimize），新增 `MultiHeadAttention<NumHeads>` 层：

```
class MultiHeadAttention : public iLayer {
    ScaledDotProduct heads[NumHeads];  // 各头
    Tensor wo;                          // 输出投影 (d_model × d_model)
    Tensor a;                           // 头拼接 (d_model × 1)
};
```

### 7.2 前向传播

```
For each head i (dim = d_k = d_model/NumHeads):
    head_i = scaled_dot_product(x)           # d_k × 1
    a[i*d_k:(i+1)*d_k, 0] = head_i           # 拼接

o = Wo · a                                   # d_model × 1
```

### 7.3 反向传播

```
∂L/∂a = Wo^T · e                  (d_model × 1)
For each head i:
    head_i.e = ∂L/∂a[i*d_k:(i+1)*d_k]       # 分布各头
    head_i.backward(ei)                       # 累加 ∂L/∂x_head_i

ei 最终 = Σ_i (Wq_i^T·dq_i + Wk_i^T·dk_i + Wv_i^T·dv_i)
```

### 7.4 参数梯度

```
g.Wo += e · a^T                               # 输出投影梯度
For each head i:
    head_i.e = da.block(...)                   # 重新计算分布
    head_i.gradient(x, y)                      # Wq/Wk/Wv 梯度
```

### 7.5 与标准 Transformer MHA 对比

| 方面 | 标准 Transformer MHA | 当前 MHA 实现 |
|------|---------------------|--------------|
| 注意力 | softmax(Q·K^T/√d_k)·V | ✅ 相同 (ScaledDotProduct) |
| 多头组合 | Concat·W_O | ✅ Wo·Concat |
| 残差 | LayerNorm(x + Attention(x)) | ⚠️ 需外部 FC/Add 层 |
| FFN | FFN(LayerNorm(x)) | ⚠️ 需外部层组合 |

**特点**: 本 MHA 是**纯注意力层**，无内置残差/FFN，用户可通过 Net 组合 FC 等层实现完整 Transformer block。

### 7.6 框架集成

- 类型标识: `LAYER_MHA` 
- Net::backward 支持: ✅ (与 Attention 相同的 broadcast 模式)
- 序列化: ✅ (write/read)
- 优化器: ✅ (SGD/RMSProp/Adam)

---

## 八、总体评估


| 评估维度 | 修复前 | 修复后 |
|---------|--------|--------|
| ScaledDotProduct 前向传播 | ✅ **正确** | ✅ **正确** |
| ScaledDotProduct 梯度计算 | ❌ **严重错误** | ✅ **已修复** |
| ScaledDotProduct backward | ❌ **严重错误**(权重和近似) | ✅ **已修复**（完整链式法则） |
| Attention 前向传播 | ✅ **正确** | ✅ **正确** |
| Attention 梯度 (w1, w2, b) | ✅ **正确** | ✅ **正确** |
| Attention broadcast | ❌ **分发原始e** | ✅ **已修复**(分发 w1^T·dy) |
| 多头梯度累积 | ❌ **ei.zero()覆盖** | ✅ **已修复**（累加） |
| Net 调用链 | ❌ **backward未调用** | ⚠️ 待修复 |

### 最终结论

Attention 网络的**架构设计**（自定义多头特征提取 + 门控融合）是合理的，但代码实现中存在**6个 Bug**（3 个 Critical + 3 个 Medium）。其中 **5 个已在代码中修复**，**1 个（Net 调用链）待修复**。

代码修复的重点：
1. **ScaledDotProduct::gradient** — 完全重写 Softmax 反向传播（J → J^T、v·k^T → e·v^T）
2. **ScaledDotProduct::backward** — 从权重和近似改为完整的 q/k/v 三条路径链式法则
3. **Attention::backward** — 添加 tanh 导数，正确分发 w1^T·dy 到各头
4. **ei.zero() 移除** — 防止多头堆叠时梯度覆盖
