# 任务进度与入口

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/optimization_report.md`](docs/optimization_report.md) | **问题与优化总记录**（35 项，按类别，含实测证据与未解决问题） |
| [`docs/convdqn.md`](docs/convdqn.md) | ConvDQN 专项 —— 含**尚未解决**的端到端问题与三条候选路线 |
| [`docs/temporal_sequence_optimization.md`](docs/temporal_sequence_optimization.md) | 时序信用分配专项（reps / resetState / BPTT / LSTM adding problem） |
| `docs/*_analysis.md`、`docs/*_design.md` | 各算法原有分析与设计文档（**部分早于本轮修复**） |

---

## 当前状态

### 已验证正常

- **干净全量构建**：无 error，26.1 s
- **测试套件**：14 项注册进 CTest，`ctest -j 6` 约 100 s
- **稳定通过的测试**：`test_rl`、`test_dqn`、`test_lstm`（12/12 × 20 次）、
  `test_ssm`、`test_mamba`、`test_moe`、`test_gradcheck`（25/25）、
  `test_convdqn`（5/5，确定性）、`test_axis`（3/3）
- **时序任务**：`test_mpg` Test 3 20/20；`test_pg` Test 3 18/20
- **端到端**：`dqn`/`dpg`/`drpg`/`mpg`/`convpg` 在真实环境中都能在 1200 步内学会
  （approach% 62→98% / 84→94% / 82→95% / 84→97% / 81→96%）

### 未解决 / 有残留

| 项 | 说明 |
|---|---|
| **ConvDQN 端到端** | 单元测试 5/5，但真实游戏中 5000 步仍学不会（approach ~50%，劣于随机 76%）。已排除奖励量级、输入平移、自举、负值、反向传播管线。见 `docs/convdqn.md` §4 |
| `test_convpg` Test 1 | 每次都失败（改动前即如此） |
| `test_trpo` Test 1 | 每次都失败（改动前即如此） |
| `test_sac` | 间歇失败（改动前即如此） |
| `test_mpg` Test 4 | 9/20 —— MPG 在上下文 bandit 上确实弱，已用配对探针**排除**是测试写法问题 |
| `test_pg` Test 4 | ~80% |
| 测试套件退出码 | 因上述 flaky 项而不可靠（`ctest` 整体约 60–85% 通过）。若要当 CI 门禁，建议**固定种子**或"多种子取多数" |
| `rl/rl.pri` | 废弃 qmake 文件，仍引用已删除的 `qlstm`；任何 qmake 构建都会坏 |
| ConvDQN 回放缓存 | `maxMemorySize=4096` ⇒ 约 435 MiB。可下调 `agent.cpp` 中的 `4096` |

---

## 构建

### 前置条件（本机已配置到用户 PATH）

| 需要 | 路径 |
|---|---|
| MSVC 2022 | `C:\Program Files\Microsoft Visual Studio\2022\Community` |
| vswhere | `C:\Program Files (x86)\Microsoft Visual Studio\Installer` |
| Ninja | `C:\Qt\Tools\Ninja` |
| Qt 6.9.2 msvc2022_64 | `C:\Qt\6.9.2\msvc2022_64\bin` |
| CMake | `C:\Qt\Tools\CMake_64` |

### 配置 + 构建

```powershell
# DSH / 非交互式进程的环境块是旧的，需要显式前置 PATH
$env:PATH = "C:\Qt\Tools\Ninja;C:\Program Files (x86)\Microsoft Visual Studio\Installer;C:\Qt\6.9.2\msvc2022_64\bin;" + $env:PATH

cd E:\home\lab\snakeAI\snakeAI
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=amd64 -host_arch=amd64 >nul && cmake -S . -B build\verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 && cmake --build build\verify"
```

### ⚠️ 头文件依赖注意事项

本机 MSVC 的中文 locale 让 CMake 记录的 `/showIncludes` 前缀与实际输出
（`注意: 包含文件:`）不匹配，ninja 曾因此**存下空的头文件依赖**。
`CMakeLists.txt` 已用 `OBJECT_DEPENDS` 显式补齐，但**在有疑虑时优先用
`--clean-first`**——否则可能一直在跑过期二进制，得到完全错误的结论。

### 可选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `SNAKEAI_BUILD_TESTS` | `ON` | 是否构建测试 |
| `SNAKEAI_USE_PCH` | `OFF` | 为 RL 库预编译 `tensor.hpp` + `layer.h`；**尚未验证**，故默认关闭 |

---

## 测试

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=amd64 -host_arch=amd64 >nul && ctest --test-dir build\verify -j 6"
```

单项直接运行（更快，且能看到详细输出）：

```powershell
build\verify\test_rl.exe         # 核心 RL 测试（最慢，~99 s）
build\verify\test_dqn.exe
build\verify\test_lstm.exe       # 12 项
build\verify\test_ssm.exe
build\verify\test_mamba.exe
build\verify\test_moe.exe
build\verify\test_mpg.exe
build\verify\test_pg.exe
build\verify\test_sac.exe
build\verify\test_trpo.exe
build\verify\test_gradcheck.exe  # 有限差分梯度校验
build\verify\test_convpg.exe
build\verify\test_convdqn.exe    # ConvDQN（本轮新增）
build\verify\test_axis.exe       # reward 统计窗口（唯一依赖 Qt 的测试）
```

> `test_axis` 是唯一链接 Qt 的测试（`AxisWidget` 是 QWidget，需要 AUTOMOC），
> 在 CTest 里通过 `ENVIRONMENT "QT_QPA_PLATFORM=offscreen"` 免窗口系统运行。

> 注意：核心测试目标名是 **`test_rl`**（源文件是 `test/test.cpp`）。
> 改名的原因是 CTest 保留目标名 `test`，`add_executable(test ...)` 会直接配置失败。

---

## 端到端基准

`bench_agent_game` 用**真实的 `Environment`**、与 GUI 完全一致的驱动方式
（`init(600,600)` → 118×118 棋盘、`setAgent(name)`、`play2` 逐步），
逐阶段输出：吃到目标数、`approach%`（朝目标靠近的走法占比）、平均蛇长、ms/步。
它**故意不注册进 CTest**（单次跑数十秒，且输出的是学习曲线而非 pass/fail）。

```powershell
build\verify\bench_agent_game.exe convdqn 2000 10   # 参数: agent 步数 阶段数
build\verify\bench_agent_game.exe astar   1200 6    # 上界: 99%
build\verify\bench_agent_game.exe rand    1200 6    # 下界: ~76%
build\verify\bench_agent_game.exe dqn     1200 6    # 62->98%
```

可选 agent：`astar` `rand` `dqn` `dpg` `drpg` `mpg` `convpg` `convdqn`
`ddpg` `ppo` `trpo` `sac` `bcq`

---

## 调试越界（AddressSanitizer）

越界问题在 Release 下表现为随机崩溃或"训练毫无效果"，很难定位。
用独立的 ASan build 目录：

```powershell
cmake -S . -B build\asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo `
      "-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi /Od"
# 把 clang_rt.asan_dynamic-x86_64.dll 拷到 exe 旁边
# 头文件依赖不可靠 -> 用 --clean-first
```

---

## 下一步（建议）

1. **ConvDQN 端到端**：按 `docs/convdqn.md` §4 的三条路线，
   优先"减少每步前传数（rollout 64→16、batch 32→16）+ 提高空间分辨率"，
   用 `bench_agent_game convdqn 2000 10` 对比（当前 0 个目标）。
2. **测试套件确定性**：为 `test_convpg` / `test_trpo` / `test_sac` / `test_mpg` Test 4
   固定种子或改"多种子取多数"，让 `ctest` 退出码可用于 CI。
3. **奖励量级**：`reward0` 的距离塑形（±167）压倒终止奖励（±1.5）。
   若要改，备选写法是 `0.5f*std::tanh(diff)`（见 `environment.cpp` 注释与
   `reward4`），但**会影响所有算法**，需要重测。
