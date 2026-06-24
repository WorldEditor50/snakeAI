# Conv2d实现分析报告

> 分析文件：`rl/conv2d.hpp`  
> 相关文件：`rl/tensor.hpp`, `rl/net.hpp`, `rl/activate.h`, `rl/ilayer.h`  
> 使用方：`rl/convpg.cpp`, `rl/convdqn.cpp`

---

## 一、总体结构

- `conv2d()` — 静态卷积函数（forward核心计算）
- `iConv2d` — 基类，存放维度参数：`inChannels, outChannels, kernelSize, stride, padding, hi, wi, ho, wo`
- `Conv2d<Fn>` — 模板卷积层，支持前向、反向、梯度、优化器
- `MaxPooling2d` — 最大池化层
- `AvgPooling2d` — 平均池化层

---

## 二、严重 Bug（必然导致训练失败或数值错误）

### Bug 1: `backward()` 中 kernel 索引维度错乱

**位置**: `Conv2d::backward()` 第 186 行

```cpp
ei(n, i, j) += kernels(n, c, i - h*stride, j - k*stride)*e(c, h, k);
//                    ^^^^^^^^
//  n 是输入通道索引（0..inChannels-1），但 kernels 的第一维是 outChannels！
```

**问题**:  
- `n` 遍历 `ei.shape[0]`（即 `inChannels`）
- `kernels` 的 shape = `(outChannels, inChannels, kernelSize, kernelSize)`
- `kernels.shape[0]` 是 `outChannels`，与 `n` 的 `inChannels` 范围可能不同
- 当 `inChannels != outChannels` 时，一定越界（segfault 或读脏数据）

**修正**: 应该交换 `n` 和 `c`：

```cpp
ei(n, i, j) += kernels(c, n, i - h*stride, j - k*stride)*e(c, h, k);
//                    ^^^^^^^^
//  c: 输出通道 (outChannels),  n: 输入通道 (inChannels)
```

**影响**: `ConvPG` 第 1 个 Conv2d 层 `inChannels=1, outChannels=4`，第 2 个 Conv2d 层 `inChannels=4, outChannels=8`，均 `inChannels != outChannels`，backward 必然越界。

---

### Bug 2: `gradient()` 中 dkernels 索引维度错乱

**位置**: `Conv2d::gradient()` 第 222–227 行

```cpp
for (int c = 0; c < dkernels.shape[1]; c++) {
    for (int h = h0; h < hn; h++) {
        for (int k = k0; k < kn; k++) {
            dkernels(n, c, h, k) += x(n, i, j)*dy(n, h, k);
            //        ^^^^^^^^                      ^^^^^^^^
            //  n 是输入通道索引 (inChannels)，但 dkernels 第一维是 outChannels
            //  dy(n,...) 也用 n 索引 dy，但 dy 第一维也是 outChannels
        }
    }
}
```

**问题**:  
1. `n` 遍历 `x.shape[0]` 即 `inChannels`，但 `dkernels(n, c, ...)` 用 `n` 作为 outChannels 维度索引
2. `dy(n, h, k)` 用输入通道索引 `n` 访问 `dy`，而 `dy` 的 shape = `(outChannels, ho, wo)`

**修正**:

```cpp
dkernels(c, n, h, k) += x(n, i, j)*dy(c, h, k);
//        ^^^^^^^^                      ^^^^^^^^
```

---

### Bug 3: MaxPooling2d 的 mask 坐标系错乱

**位置**: `MaxPooling2d::forward()` 第 366 行 和 `backward()` 第 393–395 行

**Forward 存的位置**: 对于输出位置 `(i, j)`，发现最大值在 kernel 局部偏移 `(h, k)` 时：

```cpp
mask(n, h, k) = 1;  // 存在 mask(n, h, k)，h,k 是 kernel 局部坐标 (0..kernelSize-1)
```

**Backward 读的位置**: 用输出位置索引 `(h, k)` 来读 mask：

```cpp
ei(n, i, j) += mask(n, h, k)*e(n, h, k);
//                    ^^^^^^
//  h,k 在这里是输出索引（0..ho, 0..wo）！
```

**问题**:  
- Forward 存的 `h,k` = kernel 局部偏移（范围 0..kernelSize-1）
- Backward 读的 `h,k` = 输出位置索引（范围 0..ho-1, 0..wo-1）
- 坐标系完全不同，导致梯度回传指向完全错误的位置
- 而且多个输出位置可能对应同一个 mask 位置，导致写覆盖

**修正方案**: 需要为每个输出位置 `(i, j)` 记录最大值对应的输入坐标 `(p, q)`，而不是 kernel 局部偏移：

```cpp
// 方案: mask 存 (i,j) 对应的局部偏移
mask = Tensor(outChannels, ho, wo, 2);  // 最后一维存 (h_offset, k_offset)
// forward:
mask(n, i, j, 0) = h;  // 或编码为单个整数
mask(n, i, j, 1) = k;
// backward:
ei(n, h + i*stride, k + j*stride) += e(n, i, j);  // 只将梯度传给对应位置
```

---

### Bug 4: AvgPooling2d::backward 缺少 `override` 关键字

**位置**: `AvgPooling2d::backward()` 第 457 行

```cpp
void backward(Tensor &ei)  // 缺少 override！
```

由于缺少 `override`，且基类 `iLayer` 中 `backward` 是虚函数，这一函数**不会被多态调用**。当通过基类指针调用 `backward()` 时，会调用基类的空实现，导致平均池化层的梯度不会向上一层传播。

**修正如**:

```cpp
void backward(Tensor &ei) override
```

---

### Bug 5: AvgPooling2d::backward 缺少除以 kernel 面积的缩放

平均池化 forward 计算的是 `sum/(kernelSize*kernelSize)`，因此 backward 回传梯度时需要**也除以 kernel 面积**，即：

```cpp
ei(n, i, j) += e(n, h, k)/(kernelSize*kernelSize);
```

当前实现直接累加 `e(n, h, k)`，缺乏缩放。

---

## 三、严重设计问题

### Issue 1: Bias 的 shape 定义错误

**位置**: `Conv2d` 构造函数第 121 行

```cpp
b = Tensor(outChannels, kernelSize, kernelSize);
```

Conv2d 的 bias 应该是**每个输出通道一个标量**（shape = `(outChannels,)` 或 `(outChannels, 1, 1)`），但这里创建为 `(outChannels, kernelSize, kernelSize)`，与 kernel 一样的空间大小。

导致的问题：
1. Forward 中 `op += b`：`op` 的 shape 是 `(outChannels, ho, wo)`，`b` 的 shape 是 `(outChannels, kernelSize, kernelSize)`。如果 `ho != kernelSize` 或 `wo != kernelSize`，加法会逐元素对不上，多出的维度会越界或做错。
2. Gradient 中 `g.b += dy`：`g.b` 是 `(outChannels, kernelSize, kernelSize)`，`dy` 是 `(outChannels, ho, wo)`，同样 shape 不匹配。

**修正**: bias 应为 `(outChannels, 1, 1)`，然后通过广播加到 `op` 上（或在 forward 中实现手动广播）。

---

### Issue 2: Forward 中 `o /= o.max()` 破坏输出尺度

**位置**: `Conv2d::forward()` 第 163 行

```cpp
o /= o.max();
```

这行在激活函数之后，将输出归一化到 `[0, 1]`（如果存在正值），或 `[0, -1]`（如果全负）。这是一个**非标准且非常危险**的操作：

1. **除零风险**: 如果激活后所有输出为 0（例如输入全为 0 且 bias 为 0），`o.max() = 0` 导致除以零
2. **梯度反传问题**: 这相当于在激活函数后又加了一个 `o / max(o)` 的非线性操作，但 `gradient()` 中计算的导数 `XTanh::df` 没有考虑这一操作，导致梯度计算错误
3. **破坏尺度**: 输出的数值范围完全由"批次内的最大值"决定，而非由网络参数决定，使网络学到的特征失去尺度敏感性

---

### Issue 3: 非标准激活函数 XTanh

```cpp
struct XTanh {
    inline static float f(float x) {return std::tanh(x*std::log(1 + std::exp(x)));}
    // = tanh(x * softplus(x))
};
```

这是一个非常罕见的激活函数，可能引入额外问题：
- Softplus 在大负数时接近 0，因此 `x * softplus(x)` 在大负数时为 0，`tanh(0) = 0`，没有负激活值
- 梯度中 df 需要同时传入 x 和 y，使用方式与标准激活不同，可能在泛化接口中出错
- Swish 或 Mish 等更常用的选择可能更好

---

### Issue 4: MaxPooling2d 的 `mask` 覆盖问题

在 forward 中，对每个输出位置 `(i, j)`，发现一个更大的值时设置 `mask(n, h, k) = 1`。但如果有多个位置 `(i, j)` 的 max 都在相同的 kernel 局部偏移 `(h, k)`，后一个会覆盖前一个。但更重要的是同一次 forward 对一个输出位置只能有一个 max，但 mask 没有区分不同输出位置。

如果当前输出位置的最大值在之前被另一个输出位置覆盖了，那当前输出位置虽然找到了正确值，但 mask 记录的是别的输出位置的信息。这直接导致 backward 梯度传错。

但是这个问题的严重程度不如 Bug 3，因为即使 mask 记录正确，坐标系也是错的。统一描述：整体 MaxPooling backward 实现是彻底错误的。

---

## 四、次要问题

### 1. `gradient()` 中未使用的参数 `y`

```cpp
void gradient(const Tensor &x, const Tensor &y) override
```

参数 `y` 在函数体内从未使用。

### 2. 缺少批量维度 (Batch)

所有 Conv2d 相关操作都假设 batch size = 1，不支持 mini-batch 训练。Tensor 第 0 维被用作输出/输入通道，没有 mini-batch 维度。这限制了训练效率。

### 3. h0/hn/k0/kn 边界计算的数学正确性存疑

在 backward 和 gradient 中，`h0, hn, k0, kn` 的计算意图是找对应关系，但使用了 `(i - kernelSize + 1)/stride` 这样的整数除法和 `std::ceil`/`std::floor` 混合，数学正确性难以保证。标准的做法应该是直接用卷积的逆过程（转置卷积的坐标映射）。

---

## 五、问题验证过程

构造一个简单测试用例验证 Bug 1 和 Bug 2：

- Conv2d 层: `inChannels=1, outChannels=4, kernelSize=3, stride=1, padding=0`
- 输入: `(1, 5, 5)`
- Forward 可直接计算
- Backward: `kernels(n, c, ...)` 中 `n ∈ [0, 4)`（inChannels），但 `kernels.shape[0] = 4`（outChannels），范围一致时恰好不越界，但语义错误：当 `outChannels=4, inChannels=1` 时，`kernels(1, 0, 0, 0)` 表示 output_channel=1, input_channel=0 的 kernel 元素，但本应访问 input_channel=1 的 kernel 元素（不存在），而访问了另一个 output_channel 的 kernel 值。
- 更明显的情况：`inChannels=4, outChannels=8` 时，`n ∈ [0, 4)`，`kernels.shape[0] = 8`，不会越界但取错了维度。反过来 `inChannels=8, outChannels=4` 时，`n ∈ [0, 8)`，而 `kernels.shape[0] = 4`，**必然越界 crash**。

---

## 六、总结

| 级别 | 问题 | 影响 |
|------|------|------|
| 🔴 Bug 1 | backward 中 kernel 维度索引错误 | 梯度传给网络完全错误的路径，必定导致训练发散 |
| 🔴 Bug 2 | gradient 中 kernel/激活梯度索引错误 | 权重更新方向完全错误，必定导致训练发散 |
| 🔴 Bug 3 | MaxPooling2d mask 坐标系错乱 | 池化层梯度回传指向错误位置 |
| 🔴 Bug 4 | AvgPooling2d backward 缺 override | 平均池化梯度不参与反传 |
| 🟡 Bug 5 | AvgPooling2d backward 缺缩放 | 梯度数值错误 |
| 🟡 Issue 1 | Bias shape 错误 | 加法越界或数值错误 |
| 🟡 Issue 2 | `o /= o.max()` 破坏尺度 | 梯度计算忽略该操作，导数错误 |
| ⚪ Issue 3 | 不支持 batch 训练 | 效率低 |
