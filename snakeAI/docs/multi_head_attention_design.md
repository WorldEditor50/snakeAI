# Multi-Head Attention (MHA) 设计文档

## 概述

本项目实现了一个**标准 Transformer 多头注意力层** `MultiHeadAttention<NumHeads>`，基于框架内已有组件（ScaledDotProduct、Tensor::MM、Optimize）构建。

---

## 一、架构设计

### 1.1 与现有 Attention\<N\> 的区别

| 维度 | 现有 `Attention<N>` | 新 `MultiHeadAttention<NumHeads>` |
|------|---------------------|----------------------------------|
| 注意力方式 | 特征维度自注意力 | 与标准 Transformer 一致 |
| 输出投影 | `W₁·a + W₂·x + b + tanh` | `Wo·a` (纯线性投影) |
| 激活函数 | tanh | 无 |
| 残差路径 | 内置 (W₂·x) | 无 (由外部层实现) |
| 应用场景 | 特征融合/门控 | 标准 Transformer 注意力 |

### 1.2 类结构

```cpp
template<int NumHeads>
class MultiHeadAttention : public iLayer {
    int inputDim;                      // 输入维度
    int d_k;                           // 头维度 = d_model / NumHeads
    int d_model;                       // 模型维度 = NumHeads * d_k
    Tensor wo;                         // 输出投影 (d_model × d_model)
    Tensor a;                          // 头拼接缓存 (d_model × 1)
    ScaledDotProduct heads[NumHeads];  // 各注意力头
    MHAGrad g, v, m;                   // 梯度/动量/二阶矩
};
```

### 1.3 类型注册

- 类型枚举: `LAYER_MHA`（在 `iLayer::Type` 中注册）
- Net 集成: `Net::backward` 中与 `LAYER_ATTENTION` 使用相同的 broadcast 模式

---

## 二、数学推导

### 2.1 前向传播

给定输入 x ∈ ℝ^{inputDim×1}，NumHeads = h，d_k = d_model / h：

```
For each head i ∈ {0, ..., h-1}:
    q_i = Wq_i · x              (d_k × 1)
    k_i = Wk_i · x              (d_k × 1)
    v_i = Wv_i · x              (d_k × 1)
    z_i = q_i · k_i^T / √d_k   (d_k × d_k)
    ẑ_i = softmax(z_i)          (d_k × d_k)
    head_i = ẑ_i · v_i          (d_k × 1)

a = [head₀; head₁; ...; head_{h-1}]  (d_model × 1)
o = Wo · a                             (d_model × 1)
```

#### 矩阵乘法对照

| 计算 | 代码 | 数学 |
|------|------|------|
| q_i | `ikkj(q, wq, x)` | Wq·x |
| k_i | `ikkj(k, wk, x)` | Wk·x |
| v_i | `ikkj(v, wv, x)` | Wv·x |
| z_i | `ikjk(z, q, k)` | q·k^T |
| head_i | `ikkj(o, z, v)` | ẑ·v |
| 拼接 | `a.embedding({i*d_k,0}, head_i)` | concat |
| 输出 | `ikkj(o, wo, a)` | Wo·a |

### 2.2 反向传播

给定上游梯度 e = ∂L/∂o ∈ ℝ^{d_model×1}：

```
(1) ∂L/∂a = Wo^T · e                        (d_model × 1)

For each head i:
(2) ∂L/∂head_i = ∂L/∂a[i·d_k : (i+1)·d_k]  (d_k × 1)

(3) 通过 ScaledDotProduct::backward 传播:
    d_v_i = ẑ_i^T · ∂L/∂head_i                               (d_k × 1)
    d_ẑ_i = (∂L/∂head_i) · v_i^T                             (d_k × d_k)
    vec(d_z_i) = J_i^T · vec(d_ẑ_i)                          (d_k² × 1)
    d_q_i = d_z_i · k_i / √d_k                                (d_k × 1)
    d_k_i = d_z_i^T · q_i / √d_k                              (d_k × 1)

(4) 累积到 ei:
    ei += Wq_i^T · d_q_i + Wk_i^T · d_k_i + Wv_i^T · d_v_i
```

其中 J_i ∈ ℝ^{d_k²×d_k²} 是 softmax 的 Jacobian 矩阵：
```
J_i[(p,q),(r,s)] = ∂ẑ_i[p,q] / ∂z_i[r,s] = ẑ_i[p,q] · (δ_{p,r}·δ_{q,s} - ẑ_i[r,s])
```

### 2.3 参数梯度

```
g.Wo += e · a^T                     (d_model × d_model)

For each head i:
    ∂L/∂head_i = (Wo^T · e)[i·d_k : (i+1)·d_k]
    g.Wq_i += d_q_i · x^T           (d_k × inputDim)
    g.Wk_i += d_k_i · x^T           (d_k × inputDim)
    g.Wv_i += d_v_i · x^T           (d_k × inputDim)
```

---

## 三、关键设计决策

### 3.1 无内置 tanh 激活

与 `Attention<N>` 不同，MHA 无 tanh 激活函数：

- **理由**: 标准 Transformer MHA 使用纯线性输出投影
- **代价**: 需用户自行组合 LayerNorm/残差（若需要）

### 3.2 无内置残差连接

残差连接在 Transformer 中是外围架构，非注意力层的一部分：

```
Transformer Block = Add & Norm(MHA(x)) + Add & Norm(FFN(x))
```

用户可通过 `Net` 组合实现：
```cpp
// 示例：单层 Transformer
auto mha = MultiHeadAttention<8>::_(64, 64, true);
auto fc = FC::_(64, 64, true);    // 残差后的 FFN
auto net = Net(mha, fc);
```

### 3.3 梯度累加模式

`ScaledDotProduct::backward` 使用**累加模式**（而非覆盖模式）：

- `ei` 不做 `ei.zero()`，由 `MultiHeadAttention::backward` 负责初始化
- 各头反向传播梯度时自动累加到 `ei`
- 这才是正确的多头梯度累积方式（与标准 Transformer 一致）

---

## 四、框架集成

### 4.1 支持的操作

| 操作 | 支持 | 说明 |
|------|------|------|
| 前向传播 | ✅ | Wo·Concat(heads) |
| 反向传播 | ✅ | broadcast + backward |
| 参数梯度 | ✅ | gradient |
| SGD | ✅ | 优化 |
| RMSProp | ✅ | 优化 |
| Adam | ✅ | 优化 |
| 参数裁剪 | ✅ | clamp |
| 硬拷贝 | ✅ | copyTo |
| 软更新 | ✅ | softUpdateTo (Polyak) |
| 序列化 | ✅ | write/read |

### 4.2 Net::backward 集成

```cpp
// net.hpp
} else if((layers[i-1]->type == iLayer::LAYER_ATTENTION ||
           layers[i-1]->type == iLayer::LAYER_MHA ||
           layers[i-1]->type == iLayer::LAYER_SCALEDCONCAT) &&
           layers[i]->type == iLayer::LAYER_FC) {
    layers[i]->backward(layers[i-1]->e);
    layers[i-1]->broadcast();
```

---

## 五、使用示例

### 5.1 基本使用

```cpp
#include "rl/attention.hpp"

// 创建 8 头注意力：inputDim=32, d_model=64
auto mha = RL::MultiHeadAttention<8>::_(32, 64, true);
// 实际: head_dim = 64/8 = 8

// 前向
RL::Tensor input(32, 1);
// ... fill input ...
RL::Tensor &output = mha->forward(input);

// 设置损失梯度
mha->e = loss_gradient;

// 反向传播
RL::Tensor ei(32, 1);  // 输入梯度
mha->backward(ei);

// 参数梯度计算
mha->gradient(input, output);

// 优化
mha->Adam(0.001, 0.9, 0.99, 0.999, 0.999);
```

### 5.2 Transformer Block

```cpp
// Transformer Block: MHA → FC(残差) → 激活 → FC
auto mha = RL::MultiHeadAttention<8>::_(64, 64, true);
auto fc1 = RL::FC::_(64, 256, true);   // MHA后的FFN
auto fc2 = RL::FC::_(256, 64, true);   // FFN输出投影
auto net = RL::Net(mha, fc1, fc2);
```

---

## 六、与现有 Attention\<N\> 的兼容性

- **共存**: 两个类都在 `attention.hpp` 中，互不冲突
- **共用**: 都基于 `ScaledDotProduct`（已修复完整链式法则）
- **独立**: 不同的类型枚举值，Net 可区分处理
- **选择指南**:
  - 需要标准 Transformer 注意力 → `MultiHeadAttention<N>`
  - 需要特征融合 + 门控 → `Attention<N>`
  - 需要可学习残差 + tanh → `Attention<N>`

---

## 七、近期改进方向

1. **因果掩码 (Causal Mask)**: 为自回归任务添加 mask 支持
2. **Dropout**: 在 softmax 后添加 dropout
3. **LayerNorm**: 添加内置 LayerNorm 的可选项
4. **交叉注意力**: 支持 Q/K/V 来自不同输入（Encoder-Decoder）
5. **Flash Attention 优化**: 避免构建完整的 N×N 注意力矩阵
