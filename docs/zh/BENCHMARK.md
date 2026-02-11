# ACE-Step 1.5 基准测试与性能分析指南

**Language / 语言:** [English](../en/BENCHMARK.md) | [中文](BENCHMARK.md)

---

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [测试模式](#测试模式)
- [命令行参数](#命令行参数)
- [使用示例](#使用示例)
- [理解输出](#理解输出)
- [技巧与最佳实践](#技巧与最佳实践)

---

## 概述

`profile_inference.py` 是 ACE-Step 1.5 推理的综合性能分析与基准测试工具。它可以测量端到端耗时、LLM 规划耗时、DiT 扩散耗时、VAE 解码耗时等，支持不同设备、后端和配置的组合测试。

### 支持的模式

| 模式 | 说明 |
|------|------|
| `profile` | 对单次生成进行详细的计时分析 |
| `benchmark` | 运行配置矩阵（时长 × 批量 × 思考 × 步数），输出汇总表 |
| `tier-test` | 通过 `MAX_CUDA_VRAM` 模拟不同显存大小，自动测试所有 GPU 等级 |
| `understand` | 分析 `understand_music()` API（音频 → 元数据提取） |
| `create_sample` | 分析 `create_sample()` API（灵感/简单模式） |
| `format_sample` | 分析 `format_sample()` API（标题+歌词 → 结构化元数据） |

### 支持的设备与后端

| 设备 | 参数 | 说明 |
|------|------|------|
| CUDA (NVIDIA) | `--device cuda` | 推荐，默认自动检测 |
| MPS (Apple Silicon) | `--device mps` | macOS Apple 芯片 |
| CPU | `--device cpu` | 慢，仅用于测试 |
| 自动 | `--device auto` | 自动选择最佳设备（默认） |

| LLM 后端 | 参数 | 说明 |
|-----------|------|------|
| vLLM | `--lm-backend vllm` | CUDA 上最快，推荐 NVIDIA 使用 |
| PyTorch | `--lm-backend pt` | 通用后端，所有平台可用 |
| MLX | `--lm-backend mlx` | Apple Silicon 优化 |
| 自动 | `--lm-backend auto` | 根据设备自动选择（默认） |

---

## 快速开始

```bash
# 基本分析（text2music，默认设置）
python profile_inference.py

# 启用 LLM 思考模式
python profile_inference.py --thinking

# 运行基准测试矩阵
python profile_inference.py --mode benchmark

# Apple Silicon 上测试
python profile_inference.py --device mps --lm-backend mlx

# 启用 cProfile 函数级分析
python profile_inference.py --detailed
```

---

## 测试模式

### 1. `profile` — 单次运行分析

运行单次生成，输出详细计时分解。包含可选的预热和 cProfile。

```bash
python profile_inference.py --mode profile
```

**测量内容：**
- 总耗时（端到端）
- LLM 规划耗时（token 生成、约束解码、CFG 开销）
- DiT 扩散耗时（每步和总计）
- VAE 解码耗时
- 音频保存耗时

**此模式的选项：**

| 参数 | 说明 |
|------|------|
| `--no-warmup` | 跳过预热（测量将包含编译开销） |
| `--detailed` | 启用 `cProfile` 函数级分析 |
| `--llm-debug` | 深度 LLM 调试（token 数量、吞吐量） |
| `--thinking` | 启用 LLM 思维链推理 |
| `--duration <秒>` | 覆盖音频时长 |
| `--batch-size <n>` | 覆盖批量大小 |
| `--inference-steps <n>` | 覆盖扩散步数 |

### 2. `benchmark` — 配置矩阵测试

运行配置矩阵并输出汇总表。自动适配 GPU 显存限制。

```bash
python profile_inference.py --mode benchmark
```

**默认矩阵：**
- 时长：30s, 60s, 120s, 240s（根据 GPU 显存裁剪）
- 批量大小：1, 2, 4（根据 GPU 显存裁剪）
- 思考模式：True, False
- 推理步数：8, 16

**输出示例：**

```
Duration   Batch   Think   Steps   Wall(s)    LM(s)      DiT(s)     VAE(s)     Status
--------------------------------------------------------------------------------------------------------------------------
30         1       False   8       3.21       0.45       1.89       0.52       OK
30         1       True    8       5.67       2.91       1.89       0.52       OK
60         2       False   16      12.34      0.48       9.12       1.85       OK
...
```

**保存结果为 JSON：**

```bash
python profile_inference.py --mode benchmark --benchmark-output results.json
```

### 3. `understand` — 音频理解分析

分析 `understand_music()` API，从音频 codes 提取元数据（BPM、调性、拍号、描述）。

```bash
python profile_inference.py --mode understand
python profile_inference.py --mode understand --audio-codes "your_audio_codes_string"
```

### 4. `create_sample` — 灵感模式分析

分析 `create_sample()` API，从简单文本查询生成完整歌曲蓝图。

```bash
python profile_inference.py --mode create_sample
python profile_inference.py --mode create_sample --sample-query "一首柔和的孟加拉情歌"
python profile_inference.py --mode create_sample --instrumental
```

### 5. `format_sample` — 元数据格式化分析

分析 `format_sample()` API，将描述+歌词转换为结构化元数据。

```bash
python profile_inference.py --mode format_sample
```

### 6. `tier-test` — 自动化 GPU 等级测试

使用 `MAX_CUDA_VRAM` 自动模拟不同的 GPU 显存大小，并在每个等级运行生成测试。这是修改 `acestep/gpu_config.py` 后验证所有 GPU 等级是否正常工作的推荐方式。

```bash
# 测试所有等级 (4, 6, 8, 12, 16, 20, 24 GB)
python profile_inference.py --mode tier-test

# 测试特定显存大小
python profile_inference.py --mode tier-test --tiers 6 8 16

# 启用 LM 测试（在支持的等级上）
python profile_inference.py --mode tier-test --tier-with-lm

# 快速测试：非量化等级跳过 torch.compile
python profile_inference.py --mode tier-test --tier-skip-compile
```

**每个等级验证的内容：**
- 正确的等级检测和 `GPUConfig` 构建
- 模型初始化（DiT、VAE、文本编码器，可选 LM）
- 短时间生成（30秒时长，batch=1）无 OOM 完成
- 自适应 VAE 解码回退（GPU → CPU 卸载 → 完全 CPU）
- 显存使用保持在模拟限制内

**输出示例：**

```
TIER TEST RESULTS
====================================================================================================
  VRAM    Tier       LM      Duration   Status    Peak VRAM    Notes
  ──────────────────────────────────────────────────────────────────────────────
  4GB     tier1      —       30s        ✅ OK     3.8GB        VAE 在 CPU 上解码
  6GB     tier2      —       30s        ✅ OK     5.4GB        分片 VAE chunk=256
  8GB     tier4      0.6B    30s        ✅ OK     7.2GB        vllm 后端
  12GB    tier5      1.7B    30s        ✅ OK     10.8GB       vllm 后端
  16GB    tier6a     1.7B    30s        ✅ OK     14.5GB       启用卸载
  20GB    tier6b     1.7B    30s        ✅ OK     17.2GB       无卸载
  24GB    unlimited  4B      30s        ✅ OK     21.3GB       所有模型在 GPU 上
```

> **注意**: `tier-test` 模式使用 `torch.cuda.set_per_process_memory_fraction()` 强制执行显存硬上限，即使在高端 GPU（如 A100 80GB）上也能实现真实的模拟。

#### 边界测试

使用 `--tier-boundary` 查找可以安全关闭 INT8 量化和 CPU 卸载的最低显存等级。对每个等级最多测试三种配置：

1. **default** — 等级的标准设置
2. **no-quant** — 关闭量化，卸载不变
3. **no-offload** — 不使用量化，也不使用 CPU 卸载

```bash
# 在所有等级运行边界测试
python profile_inference.py --mode tier-test --tier-boundary

# 启用 LM 的边界测试
python profile_inference.py --mode tier-test --tier-boundary --tier-with-lm

# 将边界测试结果保存为 JSON
python profile_inference.py --mode tier-test --tier-boundary --benchmark-output boundary_results.json
```

输出包含一个 **边界分析** 摘要，显示每种能力的最低等级。

#### 批次大小边界测试

使用 `--tier-batch-boundary` 查找每个等级的最大安全批次大小。对每个等级，工具会递进测试批次大小 1、2、4、8（在首次 OOM 时停止），同时测试启用 LM 和未启用 LM 的配置：

```bash
# 运行批次边界测试
python profile_inference.py --mode tier-test --tier-batch-boundary --tier-with-lm

# 测试特定等级
python profile_inference.py --mode tier-test --tier-batch-boundary --tier-with-lm --tiers 8 12 16 24
```

输出包含一个 **批次边界摘要**，显示每个等级在有 LM 和无 LM 配置下的最大成功批次大小。

---

## 命令行参数

### 设备与后端

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `auto` | 设备：`auto` / `cuda` / `mps` / `cpu` |
| `--lm-backend` | `auto` | LLM 后端：`auto` / `vllm` / `pt` / `mlx` |

### 模型路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config-path` | `acestep-v15-turbo` | DiT 模型配置 |
| `--lm-model` | `acestep-5Hz-lm-1.7B` | LLM 模型路径 |

### 硬件选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--offload-to-cpu` | 关闭 | 不使用时卸载模型到 CPU |
| `--offload-dit-to-cpu` | 关闭 | 不使用时卸载 DiT 到 CPU |
| `--quantization` | 无 | 量化：`int8_weight_only` / `fp8_weight_only` / `w8a8_dynamic` |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--duration` | 来自示例 | 音频时长（秒） |
| `--batch-size` | 来自示例 | 批量大小 |
| `--inference-steps` | 来自示例 | 扩散推理步数 |
| `--seed` | 来自示例 | 随机种子 |
| `--guidance-scale` | 7.0 | DiT 的 CFG 引导缩放 |

### LLM / CoT 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--thinking` | 关闭 | 启用 LLM 思维链推理 |
| `--use-cot-metas` | 关闭 | LLM 通过 CoT 生成音乐元数据 |
| `--use-cot-caption` | 关闭 | LLM 通过 CoT 改写/格式化描述 |
| `--use-cot-language` | 关闭 | LLM 通过 CoT 检测人声语言 |
| `--use-constrained-decoding` | 开启 | 基于 FSM 的约束解码 |
| `--no-constrained-decoding` | — | 禁用约束解码 |
| `--lm-temperature` | 0.85 | LLM 采样温度 |
| `--lm-cfg-scale` | 2.0 | LLM CFG 缩放 |

### 分析选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `profile` | 模式：`profile` / `benchmark` / `tier-test` / `understand` / `create_sample` / `format_sample` |
| `--no-warmup` | 关闭 | 跳过预热 |
| `--detailed` | 关闭 | 启用 `cProfile` 函数级分析 |
| `--llm-debug` | 关闭 | 深度 LLM 调试（token 数量、吞吐量） |
| `--benchmark-output` | 无 | 保存基准测试结果为 JSON 文件 |

### 等级测试选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tiers` | `4 6 8 12 16 20 24` | 要模拟的显存大小（GB） |
| `--tier-with-lm` | 关闭 | 在支持的等级上启用 LM 初始化 |
| `--tier-skip-compile` | 关闭 | 非量化等级跳过 `torch.compile` 以加速迭代 |
| `--tier-boundary` | 关闭 | 对每个等级测试 no-quant 和 no-offload 变体，查找最低能力边界 |
| `--tier-batch-boundary` | 关闭 | 对每个等级测试批次大小 1、2、4、8，查找最大安全批次大小 |

### 输入选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--example` | `example_05.json` | `examples/text2music/` 中的示例 JSON |
| `--task-type` | `text2music` | 任务类型：`text2music` / `cover` / `repaint` / `lego` / `extract` / `complete` |
| `--reference-audio` | 无 | 参考音频路径（用于翻唱/风格迁移） |
| `--src-audio` | 无 | 源音频路径（用于音频到音频任务） |
| `--sample-query` | 无 | `create_sample` 模式的查询文本 |
| `--instrumental` | 关闭 | 生成纯音乐（用于 `create_sample`） |
| `--audio-codes` | 无 | 音频 codes 字符串（用于 `understand` 模式） |

---

## 使用示例

### 对比不同设备

```bash
# NVIDIA GPU
python profile_inference.py --device cuda --lm-backend vllm

# Apple Silicon
python profile_inference.py --device mps --lm-backend mlx

# CPU 基线
python profile_inference.py --device cpu --lm-backend pt
```

### 对比不同 LLM 模型

```bash
# 轻量版 (0.6B)
python profile_inference.py --lm-model acestep-5Hz-lm-0.6B

# 默认版 (1.7B)
python profile_inference.py --lm-model acestep-5Hz-lm-1.7B

# 大型版 (4B)
python profile_inference.py --lm-model acestep-5Hz-lm-4B
```

### 思考模式 vs 非思考模式

```bash
# 不使用思考（更快）
python profile_inference.py --mode benchmark

# 使用思考（质量更好，更慢）
python profile_inference.py --thinking --use-cot-metas --use-cot-caption
```

### 低显存测试

```bash
# 卸载 + 量化
python profile_inference.py --offload-to-cpu --quantization int8_weight_only --lm-model acestep-5Hz-lm-0.6B
```

### 完整基准测试套件

```bash
# 运行完整基准测试矩阵并保存结果
python profile_inference.py --mode benchmark --benchmark-output benchmark_results.json

# 查看 JSON 结果
cat benchmark_results.json | python -m json.tool
```

### 函数级分析

```bash
# 启用 cProfile 进行详细的函数级分析
python profile_inference.py --detailed --llm-debug
```

---

## 理解输出

### 耗时分解

分析器会打印详细的耗时分解：

```
TIME COSTS BREAKDOWN
====================================================================================================
  Component                          Time (s)       % of Total
  ─────────────────────────────────────────────────────────────
  LLM Planning (total)               2.91           45.2%
    ├─ Token generation              2.45           38.1%
    ├─ Constrained decoding          0.31            4.8%
    └─ CFG overhead                  0.15            2.3%
  DiT Diffusion (total)              1.89           29.4%
    ├─ Per-step average              0.24            —
    └─ Steps                         8               —
  VAE Decode                         0.52            8.1%
  Audio Save                         0.12            1.9%
  Other / Overhead                   0.99           15.4%
  ─────────────────────────────────────────────────────────────
  Wall Time (total)                  6.43          100.0%
```

### 关键指标

| 指标 | 说明 |
|------|------|
| **Wall Time** | 从开始到结束的端到端耗时 |
| **LM Total Time** | LLM 规划耗时（token 生成 + 解析） |
| **DiT Total Time** | 扩散耗时（所有步骤合计） |
| **VAE Decode Time** | 将潜变量解码为音频波形的耗时 |
| **Tokens/sec** | LLM token 生成吞吐量（需 `--llm-debug`） |

---

## 技巧与最佳实践

1. **始终包含预热**（默认）— 首次运行包含 JIT 编译和内存分配开销。预热确保测量反映稳态性能。

2. **使用 `--benchmark-output`** 将结果保存为 JSON，方便后续分析或跨硬件对比。

3. **对比思考开启 vs 关闭** — 思考模式显著增加 LLM 耗时，但可能提升生成质量。

4. **使用代表性时长测试** — 短时长（30s）以 LLM 耗时为主；长时长（240s+）以 DiT 耗时为主。

5. **GPU 显存自动适配** — benchmark 模式会自动将时长和批量大小裁剪到 GPU 可处理的范围，使用 `acestep/gpu_config.py` 中的自适应等级系统。

6. **谨慎使用 `--detailed`** — `cProfile` 会增加开销；仅在需要调查函数级瓶颈时使用。

7. **使用 `tier-test` 进行回归测试** — 修改 GPU 等级配置后，运行 `--mode tier-test` 验证所有等级仍然正常工作。这在更改卸载阈值、时长限制或 LM 模型可用性时尤为重要。

8. **真实模拟低显存** — 使用 `MAX_CUDA_VRAM` 时，系统通过 `set_per_process_memory_fraction()` 强制执行显存硬上限，因此模拟期间的 OOM 错误反映了消费级 GPU 上的真实行为。
