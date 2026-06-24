# DQN 算法梯度问题分析报告

## 概述

DQN 当前使用的网络架构为：

```cpp
QMainNet = Net(
    TransformerBlock<8>::_(stateDim, true),               # 特征提取
    TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true), # 全连接 (已修正)
    Layer<Linear>::_(hiddenDim, actionDim, true, true)     # Q值输出
)
```

本文对梯度通过该三层的 **完整传递路径** 进行逐级追踪，发现并修复了4个严重梯度错误和1个系统性问题。

---

## 🔴 B1: 维度不匹配 — 越界内存访问 (UB) [已修复]

### 位置
`rl/dqn.cpp` 第34行

### 错误代码
```cpp
TanhNorm<Sigmoid>::_(8*stateDim, hiddenDim, true, true)
// TransformerBlock<8> 输出: stateDim × 1
// TanhNorm 权重: hiddenDim × 8*stateDim
// 矩阵乘法 w·x: 迭代 k: 0→8*stateDim-1, 但 x 只有 stateDim 个元素 → 越界读!
```

### 修复代码
```cpp
TanhNorm<Sigmoid>::_(stateDim, hiddenDim, true, true)
```

### 原因
`TransformerBlock<8>` 的 `d_model = stateDim`，输出为 `stateDim × 1`。`MultiHeadAttention` 中 8 个 head 的拼接通过 `Wo(stateDim × stateDim)` 投影回 `stateDim`。而 TanhNorm 被错误地设置为 `inputDim=8*stateDim`，导致权重矩阵 `w: (hiddenDim × 8*stateDim)` 在乘法 `w·x` 时越界读取 `x` 的第 `stateDim` ~ `8*stateDim-1` 个元素（未初始化内存）。

---

## 🔴 B2: Net::backward 不调用 layers[0]->backward() [已修复]

### 位置
`rl/net.hpp` 第51-86行

### 错误代码
```cpp
void backward(const Tensor &loss) {
    layers[outputIndex]->e = loss;
    for (int i = layers.size() - 1; i > 0; i--) {  // i = 2, 1
        layers[i]->backward(layers[i - 1]->e);      // 只调用 layers[2], layers[1]
    }
    // ❌ layers[0]->backward() 从未被调用!
}
```

### 具体影响
对于三层的 `Net(TransformerBlock, TanhNorm, Linear)`，执行路径为：
1. **i=2**: `Layer<Linear>::backward(layers[1]->e)` → 设置 `layers[1]->e = W₂^T · dL/dQ`
2. **i=1**: `TanhNorm::backward(layers[0]->e)` → 设置 `layers[0]->e = W₁^T · dy`
3. **终止**: `layers[0]->backward()` 不执行

这意味着 `TransformerBlock` 内部的反向传播逻辑完全跳过：
- Residual2 的梯度分叉未执行
- FFN 反向 (W_down^T, W_up^T, GeLU') 未执行
- LayerNorm2 反向 (gamma2/sig2 雅可比) 未执行
- MHA 反向 (Wo^T, 各 head 的 QKV 反向) 未执行
- LayerNorm1 反向 (gamma1/sig1 雅可比) 未执行
- LN gamma/beta 梯度累加未执行
- attn.g.wo、attn.heads[i].g.wq/wk/wv 的梯度累加未执行

### 修复代码
```cpp
void backward(const Tensor &loss) {
    layers[outputIndex]->e = loss;
    for (int i = layers.size() - 1; i > 0; i--) {
        layers[i]->backward(layers[i - 1]->e);
    }
    /* ✅ 新增: 调用 layers[0]->backward() 以执行 TransformerBlock 等复合层的内部反向传播 */
    Tensor inputGrad(layers[0]->o.totalSize, 1);
    inputGrad.zero();
    layers[0]->backward(inputGrad);
}
```

### 对 B3 的纠正
原分析报告中 B3 指出 `TransformerBlock::gradient()` 缺少 MHA/LN 梯度计算。但经过仔细阅读代码发现：

**`TransformerBlock::backward()`（第182-372行）已经包含完整的 MHA 和 LN 梯度计算:**
- **Wo 梯度**: `Tensor::MM::ikjk(attn.g.wo, d_x_res1, attn.a);`（第278行）
- **各 head 的 Wq/Wk/Wv 梯度**: `Tensor::MM::ikjk(head.g.wq, dq, x_norm1)` 等（第319-321行）
- **g2.gamma/beta 梯度**: `g2.gamma[i] += d_x_norm2[i] * x_hat; g2.beta[i] += d_x_norm2[i];`（第258-259行）
- **g1.gamma/beta 梯度**: `g1.gamma[i] += d_x_norm1[i] * x_hat; g1.beta[i] += d_x_norm1[i];`（第366-367行）

**所以 B3 不是"缺失梯度计算代码"，而是 B2 导致这些代码从未被执行。** B2 + B4 的修复使 B3 的 MHA/LN 梯度路径被重新激活。

---

## 🔴 B4: TanhNorm::backward 跳过激活函数导数 [已修复]

### 位置
`rl/layer.h` 第829-849行

### 错误代码
```cpp
void backward(Tensor &ei) override {
    Tensor::MM::kikj(ei, w, e);   // ❌ 直接 w^T · e, 跳过 sigmoid' 和 tanh'
}
```

### 问题
TanhNorm 的前向计算链：`o1 = w·x → o2 = tanh(o1·r) → o = sigmoid(o2+b)`

完整链式法则应该是：`ei = w^T · [ sigmoid'(o) · r · tanh'(o2) · e ]`

但原始实现直接 `ei = w^T · e`，缺失了 sigmoid 和 tanh 的导数。

### 修复代码
```cpp
void backward(Tensor &ei) override {
    Tensor dy(outputDim, 1);
    for (std::size_t i = 0; i < dy.totalSize; i++) {
        float d1 = Fn::df(o[i]) * e[i];       // sigmoid'(o) * dL/do
        float d2 = o2[i];
        dy[i] = r * (1 - d2 * d2) * d1;       // tanh'(o2) * r * d1
    }
    Tensor::MM::kikj(ei, w, dy);               // w^T · (完整的激活导数)
}
```

---

## 🔴 B5: Net::gradient() 迭代顺序导致使用已清零的输出 [已修复]

### 位置
`rl/net.hpp` 第87-102行

### 错误代码
```cpp
void gradient(const Tensor &x, const Tensor &y) {
    layers[0]->gradient(x, y);
    for (std::size_t i = 1; i < layers.size(); i++) {
        Tensor &out = layers[i - 1]->o;  // ❌ layers[i-1]->gradient() 已清零 o!
        layers[i]->gradient(out, y);
    }
}
```

### 问题
每层的 `gradient()` 在其结束时执行 `e.zero(); o.zero();`（见 `layer.h` 第223-225行、第847-848行等）。

正向迭代时：
1. `layers[0]->gradient(x, y)` → **清零 layers[0]->o**
2. `layers[1]->gradient(layers[0]->o, y)` → 使用已被清零的 `layers[0]->o` 作为输入 x

对于 `Layer<Linear>` 这类激活层，`gradient()` 使用 `x` 计算 `g.w += dy · x^T`。如果 `x` 是零向量，权重梯度恒为零。

### 修复代码
```cpp
void gradient(const Tensor &x, const Tensor &y) {
    for (int i = layers.size() - 1; i >= 0; i--) {  // ✅ 反向迭代
        if (i == 0) {
            layers[0]->gradient(x, y);
        } else {
            Tensor &out = layers[i - 1]->o;   // ✅ 此时 layers[i-1]->o 仍然有效
            layers[i]->gradient(out, y);
        }
    }
}
```

反向迭代确保当 `layers[i]` 使用 `layers[i-1]->o` 时，`layers[i-1]` 的 `gradient()` 尚未调用，其输出未被清零。

---

## 完整梯度追踪 (修复后)

### 理想梯度流

```
损失梯度 dL/dQ (actionDim × 1)
│
▼ [idx 2] Layer<Linear>::backward
  e₂ ← dL/dQ
  layers[1]->e = W₂^T · dL/dQ     [actionDim → hiddenDim]
│
▼ [idx 1] TanhNorm::backward
  e₁ = layers[1]->e
  dy = sigmoid'(o) · r · tanh'(o2) · e₁    [激活导数链式法则]
  layers[0]->e = W₁^T · dy                  [hiddenDim → stateDim]
│
▼ [idx 0] TransformerBlock::backward
  e₀ = layers[0]->e (已包含激活导数校正)

  Step 1: Residual 2 分叉
    d_x_res1 = e₀    (残差路径)
    d_ffn_out = e₀   (FFN路径)

  Step 2: FFN backward
    d_hidden = W_down^T · e₀
    d_hidden *= GeLU'(ffn_hidden)
    d_x_norm2 = W_up^T · d_hidden           [d_ff → d_model]

  Step 3: LayerNorm 2 backward
    g2.gamma[i] += d_x_norm2[i] · x̂₂[i]     ✓ 梯度累加
    g2.beta[i]  += d_x_norm2[i]             ✓ 梯度累加
    d_x_res1[i] += gamma2[i] · inv_sig2 ·
                   (d_x_norm2[i] - mean_dy₂ - x̂₂[i] · mean_dy_dx₂)

  Step 4: MHA backward
    da = Wo^T · d_x_res1                    [d_model × 1]
    g.wo += d_x_res1 · attn.a^T             ✓ Wo 梯度

    for each head i:
      d_head = da[dk*i : dk*(i+1)]
      dv = ẑ^T · d_head                     ✓ Wv 梯度
      dz_hat = d_head · v^T
      dz = J_softmax^T · vec(dz_hat)
      dq = dz · k / dₖ
      dk = dz^T · q / dₖ
      g.wq += dq · x_norm1^T                ✓ Wq 梯度
      g.wk += dk_t · x_norm1^T              ✓ Wk 梯度
      g.wv += dv · x_norm1^T                ✓ Wv 梯度
      d_x_norm1 += Wq^T · dq + Wk^T · dk + Wv^T · dv

  Step 5: LayerNorm 1 backward
    g1.gamma[i] += d_x_norm1[i] · x̂₁[i]     ✓ 梯度累加
    g1.beta[i]  += d_x_norm1[i]             ✓ 梯度累加
    ei[i] = d_x_res1[i] + gamma1[i] · inv_sig1 ·
            (d_x_norm1[i] - mean_dy₁ - x̂₁[i] · mean_dy_dx₁)

  输出: ei = 输入梯度 (dL/dx_input, stateDim × 1)
```

### 权重梯度计算 (after backward gradient accumulation, gradient() feeds internal sub-modules)

```
▼ gradient() 反向迭代:
  [idx 2] Layer<Linear>::gradient(prev_out, y)
    dy = Linear'(o₂) · e₂ = e₂              (Linear: df(o)=1)
    g.w += dy · x₂^T                        ✓ 权重梯度
    e₂.zero(); o₂.zero()

  [idx 1] TanhNorm::gradient(prev_out, y)
    dy = sigmoid'(o) · r · tanh'(o₂) · e    (与 backward 一致)
    g.w += dy · x₁^T                        ✓ 权重梯度
    e.zero(); o.zero()

  [idx 0] TransformerBlock::gradient(x, y)
    ffn_down.e = e
    ffn_down.gradient(ffn_hidden, y)        ✓ FFN Down 梯度
    d_hidden = W_down^T · e · GeLU'(ffn_hidden)
    ffn_up.e = d_hidden
    ffn_up.gradient(x_norm2, y)             ✓ FFN Up 梯度
    (MHA/LN 梯度的 g.wo, g.wq, g.wk, g.wv, g1.gamma/beta, g2.gamma/beta
     已在 backward() 中累加完成)
    e.zero(); o.zero(); ...缓存清零
```

---

## 修复文件清单

| # | 文件 | 修改内容 | 影响 |
|---|------|---------|------|
| B1 | `rl/dqn.cpp` | TanhNorm 输入维度 `8*stateDim` → `stateDim` | 消除越界访问 UB |
| B4 | `rl/layer.h` | TanhNorm::backward 加入 sigmoid' 和 tanh' 链式导数 | 正确反向传播激活导数到 TransformerBlock |
| B2 | `rl/net.hpp` | Net::backward 末尾添加 `layers[0]->backward(inputGrad)` | 激活 TransformerBlock 内部完整的反向路径 |
| B5 | `rl/net.hpp` | Net::gradient 从正向迭代改为反向迭代 | 使用未被清零的层输出计算权重梯度 |

---

## 遗留问题

| # | 问题 | 说明 |
|---|------|------|
| 6 | 探索率衰减过慢 (0.9999) | 1→0.1 需约23,026步，可增大衰减因子 |
| 7 | ScaledConcat 分支中 `16*4=64` 的维度假设 | 当 `stateDim=14` 时，ScaledConcat 输出 `16×4=64`，仍与 `TanhNorm::_(64, ...)` 匹配 |
| 8 | TransformerBlock::gradient 中 MHA 梯度可能重复计算 | backward() 中已通过 `Tensor::MM::ikjk(attn.g.wo, d_x_res1, attn.a)` 累加 Wo 梯度，gradient() 中若再调用 `attn.gradient()` 会重复。当前 gradient() 未调用 attn.gradient() 所以无重复问题 |
