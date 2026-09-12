# snakeAI 问题与优化总记录

本文档是**跨模块的问题清单与优化记录**，覆盖一次完整的代码审计与修复过程。
每条都注明：问题 → 影响 → 修法 → **可复现的验证证据**（或明确标注"未验证/假设"）。

配套文档：

| 文档 | 内容 |
|---|---|
| `docs/convdqn.md` | ConvDQN 专项（含**尚未解决**的端到端问题） |
| `docs/temporal_sequence_optimization.md` | 时序信用分配专项（reps / resetState / BPTT） |
| `../task_progress.md` | 当前状态、构建与测试入口 |
| `docs/*_analysis.md`、`docs/*_design.md` | 各算法原有分析（部分早于本次修复，见文末说明） |

**诚实声明**：本文档中标注 ✅ 的项目都已修复并有实测数据支撑；
标注 ⚠️ 的是**已确认但未修改**（仅记录，或修改后仍有残留问题）；
标注 ❓ 的是**未解决**的问题或尚未验证的假设。
我自己的错误判断也保留在文中（见 §3.5、§9.6）。

---

## 1. 总览

| # | 问题 | 类别 | 状态 |
|---|---|---|---|
| 1 | `flatten()` 返回一维 shape → conv→FC 反向越界读 | 崩溃/UB | ✅ |
| 2 | `Net::backward` 传给 `layers[0]` 的 `inputGrad` 是扁平的 → Conv2d 读 `shape[2]` 越界 | 崩溃/UB | ✅ |
| 3 | `Layer<Fn>::forward` 不清零输出 → 梯度与输出累积污染 | 正确性 | ✅ |
| 4 | `Optimize::SGD(w, g, lr, true)` 第 4 参是 gamma 而非 clipGrad → 每步把权重清零 | 正确性 | ✅ |
| 5 | `Conv2d::backward` 在用到 `dy` 之前就改写 `o`，且用 `e.shape[0]` 当通道数 | 梯度错误 | ✅ |
| 6 | 卷积/池化 forward 用构造参数的 h/w 而非实际输入 → 静默裁剪/越界 | 正确性 | ✅ |
| 7 | `MHA` 改写 `d_model`；头数不整除 d_model | 正确性 | ✅ |
| 8 | `PositionalEncoder` 的 `pe` 分配 / `pos` 累加 / backward 均错 | 梯度错误 | ✅ |
| 9 | `ScaledConcat`/`ScaledConcatP`/`Concat` backward 的 softmax 雅可比链断裂 | 梯度错误 | ✅ |
| 10 | `GRU` 重置门二遍扫描、`delta.g` 缺项 | 梯度错误 | ✅ |
| 11 | `SSM` 重复累加 `A·h`；`delta_.h` 未赋值；t=0 的 `h_prev` 未清零 | 梯度错误 | ✅ |
| 12 | `Mamba` `delta_h` 赋值而非累加；softplus 导数用了非标准形式；`C`/`W_in` 初始化过小 | 梯度错误/表达力 | ✅ |
| 13 | `MPG::gumbelMax/noiseAction/eGreedyAction` 每次调用回退循环状态 | 算法错误 | ✅ |
| 14 | `DRPG::action` / `MPG::action` 每次调用回退循环状态 | 算法错误 | ✅ |
| 15 | `agent.cpp` rollout 从不重置循环状态，但 `reinforce`/`reinforce1` 从 reset 重放 | 训练/推断不一致 | ✅ |
| 16 | `DPG` 用 `MOE<4,8>` 配 stateDim=2；`LN::Pre` 直接吃 2 维状态导致秩 1 坍缩 | 架构错误 | ✅ |
| 17 | `SAC` 的 `nextProb` 用引用而非拷贝；Q 目标不是全分布期望 | 算法错误 | ✅ |
| 18 | `VAE` 用新建的零张量当编码器输入梯度；`e2` 未清零 | 梯度错误 | ✅ |
| 19 | `moe.hpp` 的 `gate` 未清零 | 正确性 | ✅ |
| 20 | `Selu` 不是真正的 SELU | 正确性 | ✅ |
| 21 | `noise()` 除以零最大值；`clip` 参数名反了；`gaussian` 公式错 | 正确性 | ✅ |
| 22 | `Snake::reset` 死亡后留下断开的畸形蛇 | 崩溃/逻辑 | ✅ |
| 23 | `Environment` 构造函数整数成员未初始化 | UB | ✅ |
| 24 | GUI 绘制与训练线程竞争访问环境状态 | 数据竞争 | ✅ |
| 25 | QLSTM 严重损坏 | 算法错误 | ✅ 删除 |
| 26 | `reward0` 距离塑形量级(±167)压倒 ±1.5 终止奖励 | 奖励设计 | ⚠️ 仅记录 |
| 27 | ConvDQN：Sigmoid Q 头无法表示负值 | 算法错误 | ✅ |
| 28 | ConvDQN：探索率永不衰减 + `noiseAction` 打乱 argmax | 算法错误 | ✅ |
| 29 | ConvDQN：卷积 ~10 ns/MAC，每游戏步 192 次前传 | 性能 | ✅ |
| 30 | **ConvDQN：真实游戏中仍然学不会** | 架构/特征 | ❓ 未解决 |
| 31 | CMakeLists：头文件被当成源文件、11 个可执行文件重复编译同一批源文件、AUTOMOC 全局开启 | 构建 | ✅ |
| 32 | ninja 未记录任何头文件依赖（MSVC 中文 locale 导致 `/showIncludes` 前缀不匹配） | 构建 | ✅ |
| 33 | `ctest` 报 "No tests were found"；目标名 `test` 与 CTest 保留名冲突 | 测试设施 | ✅ |
| 34 | 多个测试自身有缺陷（越界、把噪声当信号、训练/评估条件不一致） | 测试 | ✅ |
| 35 | `rl/rl.pri` 引用了已删除的 `qlstm` | 废弃文件 | ⚠️ 仅记录 |
| 36 | **切换 agent 后 reward 统计窗口不再显示任何数据** | GUI/逻辑 | ✅ |

---

## 2. 崩溃 / 内存越界（最高优先级）

### 2.1 `Tensor_::flatten()` 返回一维形状 ✅

```cpp
// 原实现：Tensor_ x(totalSize);   // shape = {totalSize}
// 修复：  Tensor_ x(totalSize, 1); // shape = {totalSize, 1}
```

卷积层输出 `(C,H,W)` 展平后要喂给全连接层。原实现产生 **1 维** shape，而
`MM::ikjk`/`sizes[1]` 会去读 `shape[1]`——1 维 tensor 没有 `sizes[1]`，
于是每步训练都在越界读 shape 向量。

**证据**：AddressSanitizer 在 `conv2d.hpp` 的 `ikjk` 处报 heap-buffer-overflow。

### 2.2 `Net::backward` 给第 0 层的 `inputGrad` 用扁平张量 ✅

```cpp
// 之前（两代都不对）：
//   inputGrad = Tensor(layers[0]->o.totalSize, 1);   // 按输出尺寸 → kikj 越界
//   inputGrad = Tensor(x.totalSize, 1);              // 扁平 → Conv2d 读 shape[2] 越界
// 现在：
inputGrad = Tensor(x.shape);   // 复制的是网络 INPUT 的 shape
layers[0]->backward(x, inputGrad);
```

`Conv2d::backward` 会索引 `ei.shape[1]` / `ei.shape[2]`，而 `{N,1}` 没有下标 2
→ 每次训练步都读越界。ASan 定位：`heap-buffer-overflow at conv2d.hpp:223`，
分配点是 `Net::backward`。

**这条影响已发布的程序**：ConvPG 与 ConvDQN 都是卷积打头的网络，也就是说
**它们此前每一步训练都在执行未定义行为**。修好后 ASan 干净。

顺带把 `inputGrad` 提升为 public 成员，让需要"网络输入梯度"的调用方
（如 `vae.hpp`）能直接读取，而不是在缓存被清掉后再补一次 backward。

### 2.3 `Snake::reset` 留下畸形蛇 ✅

原实现只弹出被吃掉的段、然后移动 `body[0]`，导致 `body[1]`、`body[2]` 留在旧坐标，
死亡后蛇身是断开的。改成：先清掉所有旧格、清空 body、再在空闲连续 3 格上重建
（带边界保护，`x <= rows-4`）。

### 2.4 `Environment` 构造函数整数成员未初始化 ✅

`width/height/rows/cols/unitLen` 在构造函数初始化列表中缺失 → 未定义值。
`environment.cpp` 的构造函数已补齐。

---

## 3. 梯度与反向传播正确性

### 3.1 `Layer<Fn>::forward` 不清零输出 ✅

`Tensor::MM::ikkj(o, w, x)` 是**累加**语义（不清零第一个参数）。
8 个 forward 实现都在最前面加了 `o.zero()`（`LayerNorm` 另加 `o1/o2`）。

**背景（重要，容易反复踩）**：本库的 `MM::ikkj/kikj/ikjk/kijk` **全部是累加**，
且层约定是 `forward()` 自己清零输出、`backward(x, ei)` 读本层成员 `e`（上游梯度）、
写 `ei`（**输出参数**，输入梯度）。这两个约定混用极易出错。

### 3.2 `Optimize::SGD` 第 4 个参数是 gamma，不是 clipGrad ✅

```cpp
inline void SGD(Tensor &w, Tensor &dw, float lr, float gamma = 0,
                bool clipGrad = true);
```

`layer.h` 与 `conv2d.hpp` 里都写成 `Optimize::SGD(w, g.w, lr, true)`——
本意是"开启梯度裁剪"，实际把 **gamma 设成 1**，于是
`w = (1-1)*w - lr*dw = -lr*dw`，**每一步权重都被清零**。已改为 `lr`。

### 3.3 `Conv2d::backward` 的两个错误 ✅

```cpp
// 错误 1：先算 ei（用到 o），再算 dy，最后才改写 o —— 顺序反了
// 错误 2：通道数取自 e.shape[0]，而 MSE 的 df() 返回扁平张量 → 通道数被误解
```

修正后：先算 `dy = tanh'(o) ⊙ e`，再用 `dy` 算 `ei`；通道循环用 `o.shape[0]`。

### 3.4 卷积/池化 forward 用声明尺寸而非实际输入 ✅

`Conv2d::forward`、`MaxPooling2d::forward`、`AvgPooling2d::forward` 现在从
**实际输入**推导 `hi/wi/ho/wo`。原实现只信构造参数：不一致时会静默裁剪，
输入比声明小时还会读越界。对现有网络配置这是 no-op。

### 3.5 我自己的错误：误读 `Conv2d::_` 的参数顺序 ✅（已回退）

`Conv2d::_(inChannels, h, w, outChannels, kernelSize, stride, padding, bias, withgrad)`

我一度按 `(…, kernelSize, stride=1, padding=0)` 理解，把 ConvPG 的网络"修"成了
k5/s1/p0，全连接层输入从 200 变成 5832，直接崩溃。**已回退并在注释里改正**。
这条保留在文档里，因为它是"看起来很像 bug 其实不是"的典型。

---

## 4. 循环网络与时序信用分配

### 4.1 LSTM ✅

- `outputError` 原按 `hiddenDim` 分配，实际应 `outputDim`（越界/错位）。
- 细胞状态导数写成 `Tanh::df(states[t].c[i])`，应为 `Tanh::df(Tanh::f(c))`
  （`c` 是未激活的细胞状态，而 `Tanh::df` 接受的是**激活后的值**）。
- `read()` 里 `us`/`bo` 读错位置 → 保存/加载往返不一致。

### 4.2 GRU ✅

重置门需要**两遍扫描**（先算 `r`，再用 `r` 算候选隐状态）；重置门的梯度里
`delta.g` 缺项。已修正。

### 4.3 SSM ✅

重复累加 `A·h`；`delta_.h = delta.h` 应为赋值而非累加；t=0 时 `h_prev` 未清零。

### 4.4 Mamba ✅

- `delta_h[i] += delta_.h[i]`（原来是覆盖赋值，梯度被丢掉）。
- softplus 导数改用标准形式 `sigmoid(x)`（原实现是另一条曲线）。
- **初始化**：`C` 与 `W_in/b_in` 从 `U(-0.1, 0.1)` 提高到 `U(-1, 1)`。
  这是唯一一条纯"参数初始化"的改动，效果显著：

  | | 改动前 | 改动后 |
  |---|---|---|
  | 上下文效应 `max\|y(after A) − y(after B)\|` | 0.01 | **0.27 – 0.68** |

  即：原初始化下 Mamba 对历史输入几乎没有响应。

### 4.5 MPG 在采样路径上回退循环状态 ✅（本次修复）

```cpp
// 原实现：三个动作选择函数开头都有
mamba->h = mamba_h;      // mamba_h 是"上次训练后的快照"
```

后果有两条，都很严重：

1. **一次 rollout 内 Mamba 状态无法在步与步之间传递**（每一步都被重置回快照），
   于是采样出的动作**不含任何时序信息**；
2. `reinforce1` 从 `mamba->reset()` 开始重放轨迹求梯度，**被微分的分布
   ≠ 生成数据的分布**，梯度有偏。

`action()` 里同类代码此前已删除，`gumbelMax`/`noiseAction`/`eGreedyAction`
三处漏了。现在全部删除；`reinforce`/`reinforce1` 末尾那次
"保存→reset→重放→恢复" 是**合法的**，保留。

### 4.6 DRPG/MPG 的 `action()` 回退状态 ✅

`DRPG::action` 曾把 `lstm->h/c` 恢复成"上次训练前保存的值"，
`MPG::action` 同理恢复 `mamba->h`。于是**连续两次 `action()` 之间循环状态
无法传递**，2 步时序任务退化成"把 s1 映射到固定动作"，恰好落在 50% 基线。
已删除，并新增 `resetState()`（DRPG 清 `h/c`，MPG 清 `mamba_h/mamba->h`）。

### 4.7 rollout 与重放的状态不一致 ✅

`reinforce`/`reinforce1` 都从 **reset** 状态重放轨迹来做 BPTT，但
`agent.cpp` 的 rollout 从不重置循环状态——意味着**重放时的前向传播
与产生数据的策略不是同一个**。已在 `drpgAction`/`mpgAction` 的 rollout
开头加 `resetState()`。

**实测（2 步时序任务，2000 回合）**：

| 每回合是否 reset | DRPG | MPG |
|---|---|---|
| 否 | ~50% | ~50% |
| **是** | **100%** | **86.8% → 后续修复后 100%** |

并且"增加回合数"救不了失败的那些 run（2000/5000/10000 都卡在 ~50%）
→ 是**初始化依赖的双峰**，不是收敛慢。

### 4.8 LSTM 的 BPTT 本身是正确的

`test_lstm` 的 adding problem 修好后（见 §9.5）能达到 eval MSE ~0.003
（平凡预测器基线 0.0290），说明 `backwardAtTime` 里的
`δh_t = Wᵀδy + Σ Uᵀδgate_{t+1}`、`δc_t = δh⊙o⊙tanh'(c) + δc_{t+1}⊙f_{t+1}`
等递推都是对的。

---

## 5. 策略梯度算法

### 5.1 `DPG` 的网络配置 ✅

```cpp
// 原：MOE<4, 8> 配 stateDim = 2 —— 2 维模型配 8 个头
// 原：LayerNorm<Sigmoid, LN::Pre>::_(stateDim, hiddenDim) 直接吃 2 维状态
policyNet = Net(MOE<4, 2>::_(stateDim, true),
                Layer<Tanh>::_(stateDim, hiddenDim, true, true),   // ← 新增
                LayerNorm<Sigmoid, LN::Pre>::_(hiddenDim, hiddenDim, true, true),
                Layer<Softmax>::_(hiddenDim, actionDim, true, true));
```

`LN::Pre` 标准化的是它的**输入**：输入只有 2 个元素时，
`(x−mean)/std = [d/2, −d/2]`（d = x0−x1），2 维状态被压成一条方向，
后面那层投影退化成秩 1——这正是"策略周期性收敛到与状态无关的分布
（永远输出动作 0）"的原因。现在先做一次到 `hiddenDim` 的投影再标准化。

### 5.2 `reinforce` 与 `reinforce1` 的**精确**差异 ✅（重要）

游戏里**用 `reinforce`**（实测比 `reinforce1` 好，见 §10）。两者不是同一个估计量。
由 `CrossEntropy::df(out,t)[i] = -t[i]/out[i]` 与原地改写
`x[t].action[k] = p_k·A_t`：

```
reinforce()  :  Δz = η · p_k · A_t · (e_k − π)
reinforce1() :  Δz = η ·       A_t · (e_k − π)
```

`p_k` 是采样时 Gumbel-Softmax 赋给**实际执行动作**的概率。所以
`reinforce` 是**按策略对该动作的自信程度加权**的 REINFORCE：
选择接近抛硬币时步长被压小（类似隐式信赖域 / 方差抑制）。

两个实现都保留：游戏用 `reinforce`，测试钉住 `reinforce1`。
推导写在 `agent.cpp::dpgAction` 与 `rl/dpg.cpp::reinforce`，防止后人"修正"回去。

**顺带排除的干扰项**：

- α（温度）两者符号相反，但游戏里 `argmax((z+g)/α) = argmax(z+g)`，
  **α 根本不影响执行动作**，不是原因。
- `DRPG::reinforce` 缺了 DPG 有的 `alpha.clamp`，但 α 学习率是 `1e-7`，
  任意长度会话漂移约 `1e-2`，不是实际问题（已记录，未改）。

### 5.3 `reinforce1` 的改进（保留供测试用）✅

- 不再原地改写存储的 action（原实现只有在存储 action 是 one-hot 时才是精确的，
  而 `gumbelMax` 返回的是完整分布）；
- 熵用**整个分布** `H = −Σ π_i log π_i`，而不是单个动作的 `entropy(out[k])`；
- 温度梯度符号按 SAC 对偶目标 `J(α)=α(H−H_target)` 取 `H − H0`；
- `π_k` 下限保护 `1e-6`，避免 `1/π` 爆炸。

**关于 `scale = max(sd, 1)` 的更正**：我曾在注释里写它"降低方差"。
实际上 `Net::RMSProp` 总是以 `clipGrad = true` 调用 `Optimize::RMSProp`，
后者做的是 `dw /= dw.norm2()`（整张量 L2 归一化），
**任何全局缩放都会被除掉**，参数更新逐位不变。注释已在
`dpg.cpp`/`drpg.cpp`/`mpg.cpp`/`convpg.cpp` 四处更正为准确表述。

### 5.4 `SAC` ✅

- `Tensor nextProb = ...`（原来用引用，指向会被覆盖的内部缓冲）；
- Q 目标改为完整的 soft-value：`Σ_a π(a|s')·(minQ_a − α_a·log π(a|s'))`；
- α 的熵项改用 `prob_`。

### 5.5 `QLSTM` 已删除 ✅

缺陷严重（`rl/qlstm.cpp`、`rl/qlstm.h` 已删除，`agent.cpp`/`environment.cpp`
的入口一并移除）。

---

## 6. 值方法：DQN / ConvDQN

ConvDQN 的完整记录（含**未解决**部分）见 `docs/convdqn.md`。摘要：

| 缺陷 | 实测 | 修法 |
|---|---|---|
| Q 头是 `Layer<Sigmoid>` → Q ∈ (0,1)，而奖励以负为主 | 4 路空间 bandit 上**所有 Q 恒为 0.000**，正确率 1/4 | 改 `Layer<Linear>` |
| 探索率 `*=0.99999`/次 ≈ 需 23 万次才到地板 | 40,000 次后 ε 仍为 **0.67** | 改 0.9995 + 地板 0.05（**5,986 次到位**） |
| `noiseAction` 给每个 Q 加 U(0,2) 再按最大值归一 → argmax 被打乱 | — | agent 改用真正的 ε-greedy |
| 第一层卷积只输出 1 个特征图 | — | 改 4/8 通道（与本来就能工作的 ConvPG 一致） |
| 卷积 ~10 ns/MAC，每游戏步 192 次前传 | **157 ms/步** | 平坦步长 + 提外边界检查 → **38 ms/步（4.1×）**，数值逐位相同 |

**新增测试** `test/test_convdqn.cpp`（此前覆盖率为 **0**）：5 项全部通过，
包含自举路径验证 `Q(A,0)=1.092`（自举目标 0.99）、`Q(B,0)=1.096`、负值可达性。

**仍未解决**：ConvDQN 在真实游戏中依然不学（详见 `docs/convdqn.md` §未解决）。

---

## 7. 应用层：Snake / Environment / GUI

### 7.1 GUI 线程安全 ✅

`gamewidget` 增加了 `std::mutex envMutex` 与 `std::atomic<bool> isPlaying`；
`paintEvent` 在锁内对环境状态做快照；`run`/`setBlocks`/`setAgent`/`setTrainAgent`/`play1`/`play2` 加锁。
原先绘制线程与训练线程并发访问同一个 `Environment`（数据竞争）。

### 7.2 其它 ✅

- `environment.cpp`：`setBlocks` 支持减少方块；去掉调试 `cout`。
- `agent.cpp`：`supervisedAction` 用 `bpnn.RMSProp(1e-3, 0.9, 0.1)`；
  去掉热路径上的 `a.printValue()`。
- `common.cpp`：`bubbleSort` 循环条件 `j < len`（原来越界）。
- `util.hpp`：`noise()` 增加"最大值为 0"保护（原会除零产生 inf/NaN）；
  `clip(x, lo, hi)` 更名（原参数名 `(sup, inf)` 与语义相反）；修正 `gaussian`。
- `activate.h`：`Selu` 实现成真正的 SELU。

### 7.3 reward 统计窗口在切换 agent 后失效 ✅（本次修复）

**现象**：界面里切换 agent 之后，"Total reward/episode" 窗口再也看不到新数据。

**根因**（`axis.cpp`，两个缺陷叠加）：

1. 每个样本的 x 坐标来自成员 `x`——一个**只增不减**的计数器
   （`addPoint` 里 `points.append(QPointF(x, y)); x++;`），
   而"左移一格"的动作却写在 **`paintEvent()`** 里。
   **paintEvent 不是时间基准**：窗口 resize/expose/遮挡都会触发它，
   而且 Qt 会把 `addPoint()` 内的 `update()` 与 `readyForPaint` 信号带来的
   `update()` 合并——**每次重绘移动几个像素完全不可控**。
   结果是计数器一旦超过 `width()/2`，新样本就被放到可见窗口 `[−w, +w]` 之外。
2. `clearPoints()` 清空了 `points`，**但没有重置计数器 `x`**。
   而"切换 agent"正是唯一会调用它的路径
   （`GameWidget::setAgent` → `emit clearReward` → `AxisWidget::clearPoints`）。

两条叠加的后果正是用户看到的现象：切换 agent 后，新样本仍然在
`x = 5000, 5001, …` 处不断产生（数据一直在发），
但窗口只显示 `±300`，**看起来就像完全收不到数据**。

**修法**：

- 把"左移一格 + 丢弃移出窗口的样本"移到**样本到达时**（新的
  `AxisWidget::appendSample()`），新样本固定落在右边缘。
  这样横轴的含义变成"多少个样本之前"，而不是"重绘了多少次"。
- `paintEvent()` 变成**纯读取**（不再修改样本列表）。
- `clearPoints()` 清空并把计数器归零；由于定位已不再依赖任何跨清除状态，
  清除本身就已经足够。
- 顺带初始化 `timerID`（原为未初始化成员，`timerEvent()` 拿它做比较）。

**新增回归测试** `test/test_axis.cpp`（GUI 层此前**零覆盖**）：
4 项断言（新样本在窗口内、样本列表有界、清除后第一个样本立即可见、
重绘不修改样本）。测试用 `QWidget::render()` 强制触发 `paintEvent`，
因此不需要窗口系统（`QT_QPA_PLATFORM=offscreen`）。

**验证**：把 `axis.cpp` 临时改回旧行为后，测试报 4 项失败，
其中最关键的一条是
`first sample after clear is inside the window [x=5000.000000]`——
正是用户报告的失效状态（窗口只有 ±300）。改回修复后 3/3 通过、退出码 0。

> 说明：`clearReward` 与 `clearPoints` 的连接是本轮审计中我加的
> （`mainwindow.cpp` 里那句注释"was emitted on every agent switch but never
> connected"）。在那个连接之前 `clearPoints()` 根本不会被调用，
> 所以这个潜伏缺陷没有被暴露——**是我的改动把它变成了显性故障**。

### 7.4 `reward0` 的量级问题 ⚠️（仅记录，未改）

```cpp
/* 距离塑形返回的是 RAW 距离差 */
float diff = d1 - d2;
return diff;        // 118x118 棋盘上可达 ±167
```

而撞墙 / 吃到目标只有 **±1.5**。也就是说**距离塑形项压倒终止奖励两个数量级**。
README 里"reward 落在 -1 到 1"的说法与代码不符。

**为什么没改**：它定义了训练信号，策略梯度类算法只通过 advantage 使用奖励，
全局缩放会被梯度归一化除掉，所以它们不受影响（实测见 §10 全部正常学习）。
但**值方法必须回归它**——这是 ConvDQN 问题的候选原因之一（已做截断实验排除，见 `docs/convdqn.md`）。
`environment.cpp` 里的注释给了备选写法：`0.5f*std::tanh(diff)`（`reward4` 就是这么做的）。

---

## 8. 构建系统与测试基础设施

### 8.1 CMakeLists 重写 ✅

原实现把 `rl/*.h + rl/*.hpp + rl/*.cpp` 一起 glob 并追加到每个可执行文件，
且全局开启 `CMAKE_AUTOMOC`：

- 同一批 16 个翻译单元被 11 个可执行文件各编译一遍（**176 次编译调用而非 16 次**）；
- 10 个纯 C++ 测试目标也跑 AUTOMOC，产生 11 个 `*_autogen` 目录；
- 头文件被当成源文件列进构建。

现在：`RL_CORE` 静态库（纯 C++，不含 Qt）；Qt 代码生成只对 `SimpleRL` 开启；
显式源文件列表；CMake 3.16；Release 默认；`/bigobj`；`/MP` 只对非 Ninja 生成器
（Ninja 本身已并行）；`SNAKEAI_USE_PCH`（默认 OFF，未验证）；`SNAKEAI_BUILD_TESTS`。

**实测：干净全量构建 20.4 秒**，无 error。

### 8.2 头文件依赖丢失 ✅（这个坑很深）

CMake 记录的 `/showIncludes` 前缀是乱码（`濞夈劍鍓? 閸栧懎鎯堥弬鍥? :  `），
而 MSVC 在本机中文 locale 下实际输出 `注意: 包含文件:`——**不匹配**，
于是 ninja 存下了**空的头文件依赖**：改 `layer.h` 只会重建 1 个目标。

**后果**：这次审计中**早期的一批测量是在过期二进制上做的**，结论不可信。
（典型症状：同一个测试反复给出完全相同的失败，改代码"没有效果"。）

现在用 `set_property(SOURCE ... APPEND PROPERTY OBJECT_DEPENDS ${HEADERS})`
显式补上头文件依赖（改 `layer.h` 会重建 26 个目标）。
**在有疑虑时优先用 `--clean-first`。**

### 8.3 CTest 接入 ✅

`ctest` 以前直接报 **"No tests were found"**——测试可执行文件会被构建，
但没有任何 `add_test`，12 个测试只能手敲。

现在 `enable_testing()` + 每目标 `add_test` + `TIMEOUT 600`，共 **13 项**。
注意：CTest 保留目标名 `test`，因此核心测试目标改名为 **`test_rl`**
（源文件仍是 `test/test.cpp`）。

```powershell
ctest --test-dir build/verify -j 6
```

### 8.4 环境准备 ✅（本机一次性配置，已写入用户 PATH）

- `vswhere.exe` 不在 PATH → 加 `C:\Program Files (x86)\Microsoft Visual Studio\Installer`
- `ninja` 不在 PATH → 加 `C:\Qt\Tools\Ninja`
- Qt bin → `C:\Qt\6.9.2\msvc2022_64\bin`

每个由 DSH 拉起的 pwsh 环境块是旧的，因此构建命令要显式前置 PATH 并
`call VsDevCmd.bat -arch=amd64 -host_arch=amd64`。完整命令见 `../task_progress.md`。

---

## 9. 测试自身的缺陷

> 这一节单独列出：本项目的测试大量存在"看起来在验证、其实在测噪声或根本没执行"的情况。

### 9.1 `test_gradcheck`（新增）✅

新增文件，25/25 通过，用有限差分校验解析梯度。过程中发现：float32 中心差分的
**相消误差**会让比较不稳定（曾 24/25 浮动），加入 `1e-4` 的绝对噪声下限并固定
RNG 种子后稳定。

### 9.2 `test_convpg`（新增）✅

ConvPG 此前**零覆盖**。新增 bandit + 梯度方向两项。
（注：Test 1 目前仍有 flaky，见 §11。）

### 9.3 `test_convdqn`（新增）✅

ConvDQN 此前**零覆盖**——这正是 §6 那些缺陷能存活的原因。5 项覆盖：
空间 bandit、负值可达性、自举、探索调度、（固定种子 => 确定性）。

### 9.4 `test_lstm` Test 9 越界 ✅

`LSTM::backward()` 按**每时间步一个错误**索引 `cacheE`（`cacheE.size()` 必须等于
`states.size()`）。原测试在 10 步前向后只调一次 `cacheError()`，于是 BPTT 读到
`cacheE[1..9]` 越界——**访问违例（exit 0xC0000005）**。改为每步都缓存
（只有最后一步非零）。

### 9.5 `test_lstm` Test 9 训练与目标双双错误 ✅（本次修复）

原测试"adding problem"**从来没有真正跑过**（它先崩溃），修好越界后它开始运行，
但仍然是错的——原因有两个，互相独立：

1. **每个 epoch 只训练 1 个样本。** `lstm.reset()` 会清空 `cacheX`/`cacheE`，
   而 `SGD()` 正是消费它们的地方。原代码把 `lstm.SGD()` 放在 200 个序列的循环
   **之外**，于是 199 个序列的梯度被 `reset()` 丢掉，每 epoch 只对最后 1 个序列更新。
   同时 `Optimize::SGD` 是**按权重张量分别调用**的（每个张量各自归一化到单位
   L2 再走 `lr`），所以输出层每 epoch 只移动 ~`lr`，150 epoch 根本到不了需要的量级。
2. **目标不可表示。** `LSTM::feedForward` 对输出套了 `Tanh`，输出 ∈ (−1,1)；
   而目标是 [0,1] 两个数之和 ∈ [0,2]（均值 1.0）。**数学上不可能拟合**，
   eval MSE 恒在 0.65 附近——等于平凡"预测均值"解。于是断言
   "last < 0.8×first" 只能靠噪声通过或失败。

修法：逐序列更新 + 目标 ×0.4 缩进 tanh 值域 + 断言"优于平凡均值预测器"。

| | 修复前 | 修复后 |
|---|---|---|
| Test 9 失败率 | **10/20** | **0/20** |
| Eval MSE | ~0.65（≈平凡解） | **0.0014 – 0.0071**（平凡基线 0.0290） |

### 9.6 我自己的错误：6 次试验得出"完全消除双峰"的结论 ✅（已更正）

我最初用 6 次独立试验测量"每次更新重复同一序列"的效果，得到
`reps=1 → 5/6`、`reps=4 → 6/6`，就写下"完全消除双峰"。
**6 次试验根本无法区分 83% 与 100%**。改用 30 组**配对**试验（同种子 ⇒ 同初始化）后
真实数字见 §10。代码注释里已更正，并明确写出"早先的 6 次试验样本太小"。

### 9.7 `test_pg` / `test_mpg` Test 3（时序）✅

三处问题：

1. 没有按回合 `resetState()` → 上一回合的上下文埋在新回合下面（§4.7）；
2. **首次评估发生在第 400 回合之后**，于是"学得快的策略"没有提升空间，
   `trend` 检查把它判为失败；
3. 单条 2 步轨迹的梯度噪声极大（`G_1=r`、`G_0=γr`，均值基线后 advantage 只有
   ±0.05r），而第 0 步的动作与奖励无关却贡献同量级噪声。

修法：每回合 `resetState()`；ep 0 先测**未训练基线**；一次更新内重复同一序列
4 次（等价于 mini-batch REINFORCE）；`trend` 允许"已到顶"的初始化。

### 9.8 `test_pg` Test 4 训练/评估条件不一致 ✅

评估在**每次决策前**都 `resetState()`，测的是 `(s1, h=0)`——而 LSTM 训练时
从未见过这个条件（轨迹总是把 `h` 从前一个 `s0` 带过来）。
在同一份权重上做配对对比（完美配对）：

| | 旧探针（每步 reset） | 与训练一致的探针 |
|---|---|---|
| 通过 | 18/30 | **25/30** |

**7 项修好、0 项弄坏**（McNemar 精确 p ≈ 0.016）。

### 9.9 其它测试修正 ✅

`test_sac`/`test_trpo`/`test_pg` 里对 `Net::forward()` 返回值的**引用 vs 拷贝**
问题（`forward` 返回内部缓冲的引用，后续调用会覆盖它）。

---

## 10. 实测数据汇总

### 10.1 时序任务：每次更新内的重复次数（30 组**配对**试验）✅

同种子 ⇒ 同网络初始化，逐对比较；2000 回合；200 样本确定性评估；
成功 = 最终准确率 > 90%。

| 算法 | reps=1 | reps=4 | reps=8 |
|---|---|---|---|
| DRPG (LSTM) | 24/30 (80%) | **28/30 (93%)** | 29/30 (97%) |
| MPG (Mamba) | **14/30 (47%)** | **29/30 (97%)** | **30/30 (100%)** |

reps=4 vs reps=1 的配对差异：DRPG 5 修好 / 1 弄坏；MPG **16 修好 / 1 弄坏**
（McNemar 精确 p ≈ 0.0003）。

评估仍然只用**一条全新的 2 步序列**，任务本身没有被放水。

### 10.2 测试套件（每项 20 次独立运行）✅

| 测试 | 结果 |
|---|---|
| `test_lstm` | **12/12 项 × 20/20 全通过** |
| `test_mpg` Test 3（时序） | **20/20** |
| `test_pg` Test 3（时序） | **18/20** |
| `test_pg` Test 1 / 4 / 5 | 17 / 18 / 19（每 20） |
| `test_mpg` Test 1 / 4 | 15 / **9**（每 20，MPG 在 bandit 上确实弱） |
| `test_convdqn` | 5/5，确定性 |
| `test_gradcheck` | 25/25 |

### 10.3 端到端（真实 Environment，1200 步）✅

`bench_agent_game` 输出；`approach%` = 朝目标靠近的走法占比。

| agent | approach% | 吃到目标 | ms/步 |
|---|---|---|---|
| astar（上界） | 99% | 2–4/阶段 | ~0 |
| rand（下界） | ~76% | 0–1 | ~0.09 |
| **dqn（MLP）** | **62 → 98%** | 3 ✓ | 28 |
| dpg | 84 → 94% | 1–3 ✓ | 18 |
| drpg | 82 → 95% | 1–3 ✓ | 2 |
| mpg | 84 → 97% | 2–3 ✓ | 3 |
| convpg | 81 → 96% | 1–3 ✓ | 10 |
| **convdqn** | **~50%（劣于随机）** | **0** ✗ | 38 |

### 10.4 构建与测试基础设施 ✅

| | 数值 |
|---|---|
| 干净全量构建 | 20.4 s |
| `ctest -j6` 全量 | ~100 s |
| 注册测试数 | 13 |
| `test_rl` 单项 | 98.7 s（最长） |

---

## 11. 未解决问题与已知取舍

### ❓ 11.1 ConvDQN 在真实游戏中仍然学不会

已排除：奖励量级（截断 ±2 无变化）、输入表示（按蛇头居中无变化）、
自举路径（单元测试证明可用）、负值表示、卷积/全连接反向传播管线。
当前假设与硬约束见 `docs/convdqn.md`。

### ⚠️ 11.2 仍然 flaky / 失败的测试（改动前就存在）

| 测试 | 现象 |
|---|---|
| `test_convpg` Test 1 | 每次都失败（`P(a=1\|boardB)` 上不去） |
| `test_trpo` Test 1 | 每次都失败 |
| `test_sac` | 间歇失败 |
| `test_mpg` Test 4 | 9/20（MPG 在上下文 bandit 上确实弱，**已用配对探针排除是测试写法问题**） |
| `test_pg` Test 4 | ~80% |

**注意**：测试套件的退出码因此不可靠（`ctest` 整体约 60–85% 通过）。
若要把它当成 CI 门禁，建议**固定种子**或"多种子取多数"。

### ⚠️ 11.3 其它

- **`rl/rl.pri` 是废弃的 qmake 文件**，仍引用已删除的 `qlstm`，且缺
  mamba/ssm/mpg/convpg/moe/vae/trpo 等新文件。按"不删死代码"的要求保留，
  但任何 qmake 构建都会坏。
- **`environment.h` 的 `BLANK/BLOCK/TARGET/SNAKE` 宏是死代码**，且与
  `common.h` 的 `OBJ_NONE=0/OBJ_BLOCK=1/OBJ_SNAKE=2/OBJ_TARGET=4` 冲突。
  实际地图编码用的是 `OBJ_*`（全仓库无一处引用那四个宏）。
  **不要用那四个宏去读地图**：`TARGET=2` 会撞上 `OBJ_SNAKE=2`。
- **ConvDQN 的经验回放缓存约 435 MiB**（`maxMemorySize=4096`，
  每条 transition 存 2 张 118×118 float = 111 KB）。桌面程序里偏大，
  可下调 `agent.cpp` 里的 `4096`。
- **未加入学习式 state-value baseline**（critic）——教科书上 REINFORCE 方差的正解，
  但属于架构改动。
- **`LSTM::SGD/RMSProp/Adam` 按权重张量逐个调用 `Optimize::*`**，
  每个张量各自把梯度归一化到单位 L2 再走 `lr`——即"单位范数梯度下降"，
  与真实梯度大小无关。这解释了 §9.5 的现象，但也意味着**学习率语义与常规实现不同**。
- **`Agent::supervisedAction` 未注册**：`Environment::agentMethod` 里没有
  `"supervised"` 条目（`environment.cpp` 构造函数），所以这个已实现的智能体
  在运行时选不到。若要启用，加一行 `insert` 即可。
- **`Environment` 默认智能体是 `sac`**（`act = agentMethod["sac"]`）；
  而 `bench_agent_game` 的实测显示 `sac` 在本环境的 1200 步内表现最差
  （approach 42–78%，0–1 个目标），反倒是 `dqn`/`dpg` 系列学得最好。
- **`Net::Adam` 每次调用都 `alpha_ *= alpha; beta_ *= beta;`**（而 `alpha_/beta_`
  初值为 1），这不是标准 Adam 的偏差校正，实际近似于"无偏差校正的 Adam"。
  能用，但不是教科书形式。
- **`docs/` 下部分文档早于本次修复**（例如 `convdqn.md` 曾描述已经不存在的
  `Net::gradient()`）。已重写 `convdqn.md`；其余按需更新。

---

## 12. 附：构建、测试、基准

完整命令与前置条件见 `../task_progress.md`。速查：

```powershell
# PATH 前置（DSH 拉起的进程环境块是旧的）
$env:PATH = "C:\Qt\Tools\Ninja;C:\Program Files (x86)\Microsoft Visual Studio\Installer;C:\Qt\6.9.2\msvc2022_64\bin;" + $env:PATH

# 配置 + 构建
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=amd64 -host_arch=amd64 >nul && cmake -S snakeAI -B snakeAI\build\verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 && cmake --build snakeAI\build\verify"

# 测试（13 项）
cmd /c "call ... VsDevCmd.bat ... && ctest --test-dir snakeAI\build\verify -j 6"

# 端到端基准（真实环境，逐阶段学习曲线）
snakeAI\build\verify\bench_agent_game.exe convdqn 2000 10
```

AddressSanitizer 定位越界的配方（单独 build 目录）：

```powershell
cmake -S snakeAI -B snakeAI\build\asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo `
      "-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi /Od"
# 需要把 clang_rt.asan_dynamic-x86_64.dll 拷到 exe 旁边；
# 头文件依赖不可靠，改用 --clean-first。
```
