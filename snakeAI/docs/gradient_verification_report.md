# 梯度验证与代码简化报告

## 概述

对 `MOE`, `TransformerBlock`, `MultiHeadAttention`, `ScaledDotProduct` 四个模块进行了系统的梯度正确性审查和代码简化。

---

## 一、ScaledDotProduct (`rl/attention.hpp`)

### 问题：重复的注释代码
`backward()` 函数中所有梯度计算步骤在完成后又有一份完全重复的注释代码。

### 修改
删除 `backward()` 后半部分的全部注释代码（~40行）。简化后的代码保持了原有的正确梯度流：

```
o = z · v, z = softmax(q·k^T / d)
→ ∂L/∂v = z^T · e
→ ∂L/∂z = J^T · (e · v^T)，使用 O(N²) jacobian_transpose_mul
→ ∂L/∂q = (∂L/∂z) · k / d
→ ∂L/∂k = (∂L/∂z)^T · q / d
→ ∂L/∂x = Wq^T·∂L/∂q + Wk^T·∂L/∂k + Wv^T·∂L/∂v
→ g.Wq += ∂L/∂q · x^T, g.Wk += ∂L/∂k · x^T, g.Wv += ∂L/∂v · x^T
```

### 梯度正确性
- ✅ ∂L/∂v: 正确 (z^T · e)
- ✅ ∂L/∂z: 正确 (O(N²) Jacobian-vector product)
- ✅ ∂L/∂q, ∂L/∂k: 正确 (除以 √d)
- ✅ ∂L/∂x: 正确 (三个分支累加)
- ✅ 参数梯度: 正确 (ikjk 外积)
- ✅ 不含 `ei.zero()` — 为多头注意力累加设计

---

## 二、TransformerBlock (`rl/transformer.hpp`)

### 问题 1（严重）：FFN 梯度被计算两次

原 `backward()` 中存在严重 bug：

**第一次计算**（手动矩阵运算）：
```cpp
Tensor d_hidden(d_ff_, 1);
Tensor::MM::kikj(d_hidden, ffn_down.w, e);   // d_hidden = W_down^T · e
d_hidden[i] *= Gelu::df(ffn_hidden[i]);       // × GeLU'
Tensor d_x_norm2(d_model, 1);
Tensor::MM::kikj(d_x_norm2, ffn_up.w, d_hidden); // d_x_norm2 = W_up^T · d_hidden
```

**第二次计算**（通过 Layer::backward）：
```cpp
ffn_down.e = e;
ffn_down.backward(ffn_hidden, d_hidden);   // 又算一遍！
d_hidden[i] *= Gelu::df(ffn_hidden[i]);
ffn_up.e = d_hidden;
ffn_up.backward(x_norm2, d_x_norm2);       // 又算一遍！
```

结果：FFN 的激活梯度被加倍，参数梯度被加倍。LN2 backward 基于错误的 d_x_norm2 计算，进一步污染了所有后续梯度。

**修正**：合并为统一的 FFN backward 流程：
```cpp
ffn_down.e = e;
ffn_down.backward(ffn_hidden, d_hidden);   // d_hidden = W_down^T · e (Linear，无激活)
ffn_up.e = d_hidden;
ffn_up.backward(x_norm2, d_x_norm2);       // d_x_norm2 = W_up^T · [GeLU'(pre_act) · d_hidden]
```
Layer<Gelu> 的 backward 内部会正确处理 GeLU 导数，无需手动干预。

### 问题 2（性能严重）：O(N⁴) softmax Jacobian

原代码为每个注意力头构建完整的 N²×N² softmax Jacobian 矩阵：
```cpp
Tensor J = Softmax::jacobian(head.z);     // N² × N²
Tensor dz_hat_vec = dz_hat;
dz_hat_vec.reshape(dk*dk, 1);
Tensor dz_vec(dk*dk, 1);
Tensor::MM::kikj(dz_vec, J, dz_hat_vec);  // O(N⁴)
dz_vec.reshape(dk, dk);
```

**修正**：委托给 `MultiHeadAttention::backward()`，它内部使用 `Softmax::jacobian_transpose_mul()`，O(N²) 时间和 O(1) 额外内存。

### 问题 3（代码重复）：手动实现 MHA backward

原代码在 TransformerBlock 中手动实现了完整的 MHA backward（~60行），包括：
- 分配 Wo^T · e 到各 head
- 每个 head 的 dv/dz_hat/dz/dq/dk 计算
- 每个 head 的 Wq/Wk/Wv 参数梯度
- 最后 g.Wo += e · a^T

这些代码与 `MultiHeadAttention::backward()` 完全重复。

**修正**：单行委托：
```cpp
attn.e = d_x_res1;     // ∂L/∂attn_out
attn.backward(x_norm1, d_x_norm1);   // 委托给 MHA
```

### 修改后的梯度流

```
给定 e = ∂L/∂o (d_model×1):

Step 1: Residual 2
  d_x_res1 = e                 (shallow copy)

Step 2: FFN backward (统一委托)
  d_hidden = Linear.backward(ffn_hidden, e)  → W_down^T · e
  d_x_norm2 = Gelu.backward(x_norm2, d_hidden) → W_up^T · [GeLU'(pre_act) · d_hidden]
  (参数梯度自动计算并累加)

Step 3: LN2 backward
  对每个 i:
    x_hat = (x_res1[i] - mu2) / sig2
    d_x_res1[i] += gamma2[i]/sig2 · (d_x_norm2[i] - mean_dy - x_hat · mean_dy_dx)
  g2.gamma[i] += d_x_norm2[i] · x_hat
  g2.beta[i]  += d_x_norm2[i]

Step 4: MHA backward (委托给 MultiHeadAttention)
  attn.e = d_x_res1
  attn.backward(x_norm1, d_x_norm1)
  
  MultiHeadAttention::backward 内部:
    da = Wo^T · d_x_res1
    对每个 head i:
      heads[i].e = block(da, i*d_k, d_k)
      heads[i].backward(x_norm1, d_x_norm1)  // 累加进 d_x_norm1
    
    // head_i.backward:
    //   dv_i = z_i^T · e_i
    //   dz_hat_i = e_i · v_i^T
    //   dz_i = J^T · dz_hat_i (O(N²))
    //   dq_i = dz_i · k_i / √d_k
    //   dk_i = dz_i^T · q_i / √d_k
    //   d_x_norm1 += Wq_i^T·dq_i + Wk_i^T·dk_i + Wv_i^T·dv_i
    //   g.Wq_i += dq_i · x^T, ...
    
    g.Wo += d_x_res1 · a^T

Step 5: LN1 backward + 累加至 ei
  对每个 i:
    x_hat = (x_orig[i] - mu1) / sig1
    ei[i] += d_x_res1[i]                     // residual 路径
    ei[i] += gamma1[i]/sig1 · (d_x_norm1[i] - mean_dy - x_hat · mean_dy_dx)  // LN 路径
  g1.gamma[i] += d_x_norm1[i] · x_hat
  g1.beta[i]  += d_x_norm1[i]
```

### 梯度正确性验证
| 分支 | 路径 | 正确性 |
|------|------|--------|
| Residual 2 | d_x_res1 = e | ✅ |
| FFN Down | d_hidden = W_down^T · e | ✅ Linear::backward |
| FFN Up + GeLU | d_x_norm2 = W_up^T · [GeLU'(pre_act) · d_hidden] | ✅ Gelu.backward |
| LN2 | d_x_res1 += gamma2/sig2 · (dy - mean_dy - x̂ · mean_dy_dx̂) | ✅ |
| MHA | 多头注意力 backward | ✅ 委托给 MultiHeadAttention |
| LN1 | ei += d_x_res1 + gamma1/sig1 · (dy - mean_dy - x̂ · mean_dy_dx̂) | ✅ |
| g2.gamma/g2.beta | d_x_norm2[i] · x̂[i] / d_x_norm2[i] | ✅ |
| g1.gamma/g1.beta | d_x_norm1[i] · x̂[i] / d_x_norm1[i] | ✅ |
| g.Wo | d_x_res1 · a^T | ✅ |
| g.Wq_i/g.Wk_i/g.Wv_i | 各 head 参数梯度 | ✅ head_i.gradient |
| g.W_down/g.W_up | FFN 参数梯度 | ✅ ffn_down/up.backward |

### 复杂度改善
| 指标 | 原代码 | 新代码 |
|------|--------|--------|
| 代码行数 (backward) | ~190行 | ~70行 |
| Softmax Jacobian | O(N⁴) 时间 + O(N⁴) 内存 | O(N²) 时间 + O(1) 内存 |
| 重复计算 | FFN 梯度计算 2 次 | 统一 1 次 |
| MHA backward | 手动实现(重复) | 委托调用(DRY) |

---

## 三、MOE (`rl/moe.hpp`)

### 问题：重复的注释代码
`backward()` 函数中所有梯度计算步骤在完成后又有一份完全重复的注释代码（~20行）。

### 修改
删除 `backward()` 后半部分的全部注释代码。简化后的代码保持了原有的正确梯度流：

```
o = Σ_i gate[i] · expert_out[i]

给定 e = ∂L/∂o (d_model×1):

1. Gating 路径:
   d_gate[i] = e · expert_out[i]            (标量点积)
   d_gate_logit = J^T · d_gate              (softmax Jacobian-vector product)
   ei += Wg^T · d_gate_logit                (门控对输入的梯度)

2. Expert 路径:
   for each expert i:
     experts[i].e = gate[i] · e             (分布梯度)
     experts[i].backward(x, ei)             (累加对输入的梯度)

3. 参数梯度:
   g.wg += d_gate_logit · x^T
   g.b  += d_gate_logit
```

### 梯度正确性
- ✅ ∂L/∂expert_out[i] = gate[i] · e — 正确处理门控权重缩放
- ✅ ∂L/∂gate[i] = e · expert_out[i] — 正确的标量点积
- ✅ ∂L/∂gate_logit = J^T · ∂L/∂gate — 正确通过 softmax 回传
- ✅ ∂L/∂x = Wg^T · ∂L/∂gate_logit + Σ_i expert_i.backward(...)
- ✅ 参数梯度: g.wg += d_gate_logit · x^T, g.b += d_gate_logit

---

## 四、测试结果

| 测试 | 结果 | 说明 |
|------|------|------|
| MOE 8/8 | ✅ 全部通过 | 包含梯度流、学习、序列化等 |
| LSTM 7/7 | ✅ 全部通过 | 回归通过 |
| SSM 全部 | ✅ 全部通过 | 回归通过 |
| Mamba 全部 | ✅ 全部通过 | 回归通过 |
| SAC 2/5 | ⚠️ 预先存在失败 | alpha 退火相关问题(非本次修改导致) |
| MPG 0/4 | ⚠️ 预先存在失败 | 与策略梯度相关(非本次修改导致) |
| TRPO | ⚠️ 预先存在挂起 | 非本次修改导致 |

---

## 五、总结

| 模块 | 问题数 | 严重程度 | 修复方式 |
|------|--------|----------|----------|
| ScaledDotProduct::backward() | 1 | 低(注释) | 删除重复注释代码 |
| TransformerBlock::backward() | 3 | 严重(梯度加倍)、性能(O(N⁴))、重复 | FFN 统一委托、MHA 委托、O(N²) Jacobian |
| MOE::backward() | 1 | 低(注释) | 删除重复注释代码 |

所有修复保持原有数学正确性，消除冗余，改进性能，并降低了约 60% 的 backward 代码量。
