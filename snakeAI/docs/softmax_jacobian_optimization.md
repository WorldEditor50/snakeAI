# Softmax Jacobian 梯度优化分析

## 概述

本文档分析 `ScaledDotProduct` 中 softmax 梯度计算存在的效率问题，并给出使用 `jacobian_transpose_mul` 实现的 **O(N²) 时间 + O(1) 内存** 的优化方案。

---

## 一、问题背景

### 1.1 Softmax 定义

对于输入向量 x ∈ ℝ^N，softmax 输出为：

```
y_i = exp(x_i) / Σ_j exp(x_j)
```

### 1.2 Jacobian 矩阵

softmax 的 Jacobian 矩阵 J ∈ ℝ^(N×N):

```
J(i,j) = ∂y_i/∂x_j = y_i · (δ_ij - y_j)
```

其中 δ_ij 是 Kronecker delta。

### 1.3 原始实现的问题

在原始 `ScaledDotProduct::backward` 和 `ScaledDotProduct::gradient` 中：

```cpp
// ❌ 原始实现：显式构造 N²×N² Jacobian 矩阵
Tensor J = Softmax::jacobian(z);           // N² × N² 矩阵！
Tensor dz_hat_vec = dz_hat;                 // copy
dz_hat_vec.reshape(N*N, 1);
Tensor dz_vec(N*N, 1);
Tensor::MM::kikj(dz_vec, J, dz_hat_vec);   // J^T · vec(dz_hat)
dz_vec.reshape(N, N);
```

**问题**：
- 对于 N×N 的注意力矩阵 z，softmax 操作的输出是 N² 个元素
- Jacobian 矩阵维度为 N² × N² = N⁴ 个元素
- **时间复杂度**: O(N⁴) — 构造 N²×N² 矩阵需要 N⁴ 次操作
- **空间复杂度**: O(N⁴) — 需要存储完整的 N²×N² 矩阵

| N  | N² (softmax 元素数) | N⁴ (Jacobian 元素数) | Jacobian 内存 (float32) |
|---|---|---|---|
| 8  | 64     | 4,096       | 16 KB    |
| 16 | 256    | 65,536      | 256 KB   |
| 32 | 1,024  | 1,048,576   | 4 MB     |
| 64 | 4,096  | 16,777,216  | 64 MB    |
| 128| 16,384 | 268,435,456 | 1 GB     |

对于标准 Transformer，d_k=64 或 d_k=128 时问题已经非常严重。

---

## 二、数学本质：Jacobian-向量乘积的化简

### 2.1 核心观察

在 ScaledDotProduct 中，我们实际上不需要完整的 Jacobian 矩阵 J，而只需要 **J^T 与某个向量的乘积**。

对于 softmax 的 Jacobian J (N×N)，需要计算：

```
(J^T · v)[i] = Σ_j J(j,i) · v[j]
             = Σ_j y_j · (δ_ji - y_i) · v[j]
             = y_i · v[i] - y_i · Σ_j y_j · v[j]
             = y_i · (v[i] - dot(y, v))
```

其中 `dot(y, v) = Σ_j y_j · v_j` 是一个标量。

**关键**：J^T = J（对称矩阵），所以 J^T · v = J · v。

### 2.2 用于 ScaledDotProduct

在 ScaledDotProduct 中，z 是 N×N 注意力矩阵，softmax 作用于整个矩阵（flat 模式）。

为计算 `vec(∂L/∂z) = J^T · vec(∂L/∂ẑ)`：

```
dz[p,q] = z[p,q] · (dz_hat[p,q] - Σ_{r,s} z[r,s] · dz_hat[r,s])
```

这本质上是对 N² 个元素的一次 softmax Jacobian-向量乘积。

---

## 三、优化实现

### 3.1 新 API: `Softmax::jacobian_transpose_mul`

```cpp
struct Softmax {
    /*
        Compute J^T · v efficiently for arbitrary-shaped tensors.

        J[i,j] = y[i]·(δ_ij - y[j])  (treats y and v as flat)

        (J^T · v)[i] = y[i] · (v[i] - Σ_k y[k]·v[k])

        Time:  O(N) where N = totalSize
        Memory: O(1) extra
    */
    inline static void jacobian_transpose_mul(
        const Tensor &y, const Tensor &v, Tensor &out)
    {
        float dot = 0;
        for (std::size_t k = 0; k < y.totalSize; k++) {
            dot += y[k] * v[k];
        }
        for (std::size_t i = 0; i < y.totalSize; i++) {
            out[i] = y[i] * (v[i] - dot);
        }
    }
};
```

### 3.2 在 ScaledDotProduct 中的使用

**backward()** 中替换：

```cpp
// ❌ 原始：O(N⁴) 时间 + O(N⁴) 内存
Tensor J = Softmax::jacobian(z);
Tensor dz_hat_vec = dz_hat;
dz_hat_vec.reshape(N*N, 1);
Tensor dz_vec(N*N, 1);
Tensor::MM::kikj(dz_vec, J, dz_hat_vec);
dz_vec.reshape(N, N);

// ✅ 优化后：O(N²) 时间 + O(1) 额外内存
Tensor dz_vec(N, N);
Softmax::jacobian_transpose_mul(z, dz_hat, dz_vec);
```

**gradient()** 中同理。

---

## 四、性能对比

### 4.1 理论分析

| 指标 | 原始实现 | 优化实现 | 加速比 |
|------|---------|---------|--------|
| 时间复杂度 | O(N⁴) | O(N²) | N² 倍 |
| 额外空间复杂度 | O(N⁴) | O(1) | N⁴ 倍 |
| Jacobian 构造 | 双重循环 N²×N² | 两遍线性扫描 | — |
| 矩阵乘法 | J^T · vec(dz_hat): O(N⁴) | 无 | — |

### 4.2 实际数据

| N  | 原始: Jacobian 构造 | 原始: J^T·v (kikj) | 优化: 两个线性扫描 |
|---|---|---|---|
| 8  | 4,096 次操作 | 4,096 次操作 | 128 次操作 |
| 16 | 65,536 次操作 | 65,536 次操作 | 512 次操作 |
| 32 | 1,048,576 次操作 | 1,048,576 次操作 | 2,048 次操作 |
| 64 | 16,777,216 次操作 | 16,777,216 次操作 | 8,192 次操作 |
| 128| 268,435,456 次操作 | 268,435,456 次操作 | 32,768 次操作 |

标准 Transformer 中 d_k=64，每个头注意力矩阵 64×64=4,096 元素：
- 原始: 构造 Jacobian 1,680 万次操作 + 矩阵乘法 1,680 万次操作 ≈ 3,360 万次
- 优化: 两次线性扫描各 4,096 次 ≈ 8,192 次
- **加速约 4,000 倍**

---

## 五、原始实现的数学正确性

**结论：原始实现数学上完全正确，问题仅在于计算效率。**

### 5.1 原始 backward 的数学链

```
原始实现:
  dv   = z^T · e                               ✅ Step 1: ∂L/∂v
  dẑ   = e · v^T                                ✅ Step 2: ∂L/∂ẑ
  J    = Softmax::jacobian(z)                   ✅ Step 3a: 构造 Jacobian
  dẑ_vec = flatten(dẑ)                         ✅ Step 3b: 展平
  dz_vec = kikj(J, dẑ_vec) = J^T · dẑ_vec     ✅ Step 3c: J^T · v
  dz   = reshape(dz_vec)                       ✅ Step 3d: 恢复形状
  dq   = dz · k / d                             ✅ Step 4: ∂L/∂q
  dk   = dz^T · q / d                           ✅ Step 5: ∂L/∂k
  ei  += Wq^T·dq + Wk^T·dk + Wv^T·dv           ✅ Step 6: ∂L/∂x
```

### 5.2 kikj 计算的精确验证

`kikj` 的语义是 `result(i) = Σ_k x1(k,i) · x2(k)`。将 J = Softmax::jacobian(z) 代入：

```
J(k, i) = ẑ[k] · (δ_ki - ẑ[i])

dz_vec(i) = Σ_k J(k, i) · dẑ_vec(k)
          = Σ_k ẑ[k] · (δ_ki - ẑ[i]) · dẑ_vec(k)
          = ẑ[i] · dẑ_vec(i) - ẑ[i] · Σ_k ẑ[k] · dẑ_vec(k)
          = ẑ[i] · (dẑ_vec(i) - dot(ẑ, dẑ_vec))
```

这正是 `J^T · v` 的正确结果。✅

### 5.3 问题不在数学，在效率

| 方面 | 原始实现 | 优化实现 |
|------|---------|---------|
| **数学正确性** | ✅ 完全正确 | ✅ 完全正确 |
| **使用的公式** | `J^T · v` (通用矩阵-向量) | `y ⊙ (v - dot(y,v)·1)` (代数化简) |
| **时间复杂度** | ❌ O(N⁴) — 构造 N²×N² 矩阵 + 矩阵乘 | ✅ O(N²) — 两次线性扫描 |
| **空间复杂度** | ❌ O(N⁴) — N²×N² = 64 MB (d_k=64) | ✅ O(1) — 无需额外大内存 |

原始实现相当于：**先打印出整个九九乘法表（100个格子），再查表计算 3×5=15**。结果是对的，但做了大量无用功。

---

## 六、正确性验证

### 6.1 数学等价性

```
原始: dz = vec⁻¹(J^T · vec(dz_hat))
       J(i,j) = y[i]·(δ_ij - y[j])
       (J^T · v)[i] = Σ_j J(j,i) · v[j]
                    = Σ_j y[j]·(δ_ji - y[i]) · v[j]
                    = y[i]·v[i] - y[i]·Σ_j y[j]·v[j]
                    = y[i] · (v[i] - dot(y, v))

优化: dz = y ⊙ (dz_hat - dot(y, dz_hat)·1)
```

二者数学上完全等价。

### 6.2 数值稳定性

优化实现与原始实现使用相同的浮点操作，没有引入额外的数值误差：
- `dot = Σ y[k]·v[k]`：与原始 Jacobian 中的 `y[i]·y[j]` 使用相同的乘加操作
- `out[i] = y[i]·(v[i] - dot)`：两个浮点运算

唯一区别是原始实现了完整 Jacobian（可能被其他地方使用），优化实现直接计算结果。

---

## 七、代码变更总结

### 7.1 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `rl/activate.h` | 新增 `Softmax::jacobian_transpose_mul()` 方法 |
| `rl/attention.hpp` | `ScaledDotProduct::backward()` 和 `ScaledDotProduct::gradient()` 使用新方法 |

### 7.2 保留的 API

为保持向后兼容，原始 `Softmax::jacobian()` 方法保留未删除。新 API `jacobian_transpose_mul()` 是推荐的替代方案。

### 7.3 编译状态

✅ 编译通过，无新增警告。
