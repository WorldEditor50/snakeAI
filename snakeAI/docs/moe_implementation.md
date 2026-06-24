# MOE（Mixture of Experts）实现原理

## 概述

MoE（Mixture of Experts，混合专家模型）是一种通过门控机制（Gating Network）将输入分配到多个专家网络（Experts）分别处理，再将专家输出加权融合的架构。本实现中，每个专家是一个完整的 `TransformerBlock`（包含 Multi-Head Self-Attention 和 FFN），使得每个专家都具有强大的序列建模能力。

## 核心公式

### 1. 门控（Gating）

输入向量 $x \in \mathbb{R}^{d_{\text{model}}}$，经过线性变换和 Softmax 得到各专家的权重：

$$h = W_g \cdot x + b, \quad W_g \in \mathbb{R}^{n_{\text{experts}} \times d_{\text{model}}}$$

$$\text{gate}_i = \frac{\exp(h_i)}{\sum_{j=1}^{n_{\text{experts}}} \exp(h_j)}$$

其中 $\sum_{i} \text{gate}_i = 1$，$0 < \text{gate}_i < 1$。

### 2. 专家输出

每个专家 $E_i$ 是一个 `TransformerBlock`，对输入 $x$ 独立计算：

$$y_i = E_i(x) \in \mathbb{R}^{d_{\text{model}}}$$

TransformerBlock 内部包含：
- LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → FFN → Residual

### 3. 加权融合

最终输出为所有专家输出的加权和：

$$y = \sum_{i=1}^{n_{\text{experts}}} \text{gate}_i \cdot y_i$$

## 梯度计算

### 正向传播（forward）

```
输入 x
对于每个专家 i:
    y_i = experts[i].forward(x)        # TransformerBlock forward
gate = softmax(Wg * x + b)             # 门控权重
输出 y = Σ gate[i] * y_i               # 加权融合
```

### 反向传播（backward）

梯度通过反向传播流经三个部分：

1. **到门控网络**：
   $$\frac{\partial L}{\partial W_g} = \frac{\partial L}{\partial \text{gate}} \cdot \frac{\partial \text{gate}}{\partial (W_g x + b)} \cdot x^T$$

2. **到每个专家**（通过 gate[i] 加权）：
   $$\frac{\partial L}{\partial y_i} = \text{gate}[i] \cdot e \quad \text{(其中 e 为输出误差)}$$

3. **到输入**（ei 积累梯度）：
   $$e_{\text{input}} = \sum_i \text{gate}[i] \cdot e_{y_i} + \frac{\partial L}{\partial \text{gate}} \cdot W_g^T$$

## 网络结构

```
MOE<NumExperts, NumHeads>
├── wg: Linear(num_experts × d_model)     # 门控权重矩阵
├── b:  bias(num_experts)                  # 门控偏置
├── gate[num_experts]: Tensor              # Softmax 门控输出
├── experts[NumExperts]:                    # 专家列表
│   └── TransformerBlock<NumHeads, 0>      # 每个专家 = 完整 Transformer
│       ├── ln1: LayerNorm
│       ├── attn: MultiHeadAttention
│       │   ├── wo: Linear
│       │   ├── wq, wk, wv: Linear
│       │   └── ...
│       ├── ln2: LayerNorm
│       ├── ffn_up: Linear(d_model × d_ff)
│       ├── ffn_down: Linear(d_ff × d_model)
│       ├── gamma1: LayerNorm scale
│       ├── beta1: LayerNorm shift
│       └── ...
├── e[d_model]: Tensor                     # 输出误差（forward后设置）
├── g: {wg, b}                             # 门控梯度
└── (experts内部各自维护各自的g梯度)
```

## 关键特征

### 1. 软门控（Soft Gating）

与硬门控（Top-k routing）不同，本实现使用**软门控**——所有专家都参与计算（无 Top-k 稀疏化）。优点：
- 所有专家始终接收梯度更新，避免"死专家"问题
- 梯度流更稳定，训练更平滑

### 2. 专家多样性

每个专家作为独立 TransformerBlock，拥有**完整参数副本**（注意力权重 + FFN 权重 + LayerNorm 参数）。这带来：
- 专家可以学习不同的特征表示
- 门控自动学习分配不同输入给不同专家

### 3. 参数共享

通过模板参数 `MOE<NumExperts, NumHeads>` 在编译期确定专家数和注意力头数：
- NumExperts：专家数量（决定门控输出维度）
- NumHeads：每个专家内 MHA 的头数

## 训练流程

```
每个训练步骤:
1. out = moe.forward(x)                        # 前向传播
2. loss = MSE(out, target)                      # 计算损失
3. moe.e[j] = d(Loss)/d(out[j])                # 设置输出误差
4. moe.backward(ei)                             # 反向传播（MHA + Transformer）
5. moe.gradient(x, x)                           # 梯度计算（门控 + FFN）
6. moe.RMSProp(lr, rho, 0, true)               # 参数更新
   （或 SGD/AdamW）
```

## 门控分析

门控网络 `gate` 的行为揭示了模型的**输入路由策略**：

- 当某一输入被高概率分配到一个专家时，说明该输入"擅长"由该专家处理
- 当所有专家概率接近均匀分布时，说明输入没有明显的领域特异性
- 训练过程中 gate 分布会逐渐从均匀分布向**专业化**方向演化

## 与标准 Transformer 对比

| 特性 | 标准 Transformer | MOE + TransformerBlock |
|------|-----------------|------------------------|
| 参数量 | 1 倍 | NumExperts 倍 |
| 计算量 | 1 倍 | NumExperts 倍（无加速时） |
| 表达能力 | 单一表示空间 | 多表示空间融合 |
| 过拟合风险 | 中等 | 更高（需正则化） |
| 训练稳定 | 稳定 | 可能门控震荡 |

## 测试验证

现有 8 个测试验证了 MOE 的核心功能：

1. **前向形状测试**：输出 shape = [d_model, 1] ✓
2. **门控求和测试**：gate 概率和 = 1.0 ± 1e-4 ✓
3. **门控梯度测试**：梯度经 `gradient()` 流到 `g.wg` 和 `g.b` ✓
4. **专家梯度测试**：梯度经 `gradient()` 流到各专家内部 ✓
5. **学习能力测试**：MSE 训练损失下降 > 50% ✓
6. **保存/加载测试**：参数序列化/反序列化一致性 ✓
7. **复制/软更新测试**：`copyTo()` 和 `softUpdateTo()` 正确 ✓
8. **反向积累测试**：`backward()` 正确积累输入梯度 ✓
