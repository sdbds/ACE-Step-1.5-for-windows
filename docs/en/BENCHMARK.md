# ACE-Step 1.5 Benchmark & Profiling Guide

**Language / 语言:** [English](BENCHMARK.md) | [中文](../zh/BENCHMARK.md)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Profiling Modes](#profiling-modes)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [Understanding Output](#understanding-output)
- [Tips & Best Practices](#tips--best-practices)

---

## Overview

`profile_inference.py` is a comprehensive profiling & benchmarking tool for ACE-Step 1.5 inference. It measures end-to-end wall time, LLM planning time, DiT diffusion time, VAE decoding time, and more — across different devices, backends, and configurations.

### Supported Modes

| Mode | Description |
|------|-------------|
| `profile` | Profile a single generation run with detailed timing breakdown |
| `benchmark` | Run a matrix of configurations (duration × batch × thinking × steps) and produce a summary table |
| `tier-test` | Automatically test all GPU tiers by simulating different VRAM sizes via `MAX_CUDA_VRAM` |
| `understand` | Profile the `understand_music()` API (audio → metadata extraction) |
| `create_sample` | Profile the `create_sample()` API (inspiration / simple mode) |
| `format_sample` | Profile the `format_sample()` API (caption + lyrics → structured metadata) |

### Supported Devices & Backends

| Device | Flag | Notes |
|--------|------|-------|
| CUDA (NVIDIA) | `--device cuda` | Recommended. Auto-detected by default |
| MPS (Apple Silicon) | `--device mps` | macOS with Apple Silicon |
| CPU | `--device cpu` | Slow, for testing only |
| Auto | `--device auto` | Automatically selects best available (default) |

| LLM Backend | Flag | Notes |
|-------------|------|-------|
| vLLM | `--lm-backend vllm` | Fastest on CUDA, recommended for NVIDIA |
| PyTorch | `--lm-backend pt` | Universal fallback, works everywhere |
| MLX | `--lm-backend mlx` | Optimized for Apple Silicon |
| Auto | `--lm-backend auto` | Selects best backend for device (default) |

---

## Quick Start

```bash
# Basic profile (text2music, default settings)
python profile_inference.py

# Profile with LLM thinking enabled
python profile_inference.py --thinking

# Run benchmark matrix
python profile_inference.py --mode benchmark

# Profile on Apple Silicon
python profile_inference.py --device mps --lm-backend mlx

# Profile with cProfile function-level analysis
python profile_inference.py --detailed
```

---

## Profiling Modes

### 1. `profile` — Single Run Profiling

Runs a single generation with detailed timing breakdown. Includes optional warmup and cProfile.

```bash
python profile_inference.py --mode profile
```

**What it measures:**
- Total wall time (end-to-end)
- LLM planning time (token generation, constrained decoding, CFG overhead)
- DiT diffusion time (per-step and total)
- VAE decode time
- Audio save time

**Options for this mode:**

| Flag | Description |
|------|-------------|
| `--no-warmup` | Skip warmup run (includes compilation overhead in measurement) |
| `--detailed` | Enable `cProfile` function-level analysis |
| `--llm-debug` | Deep LLM debugging (token count, throughput) |
| `--thinking` | Enable LLM Chain-of-Thought reasoning |
| `--duration <sec>` | Override audio duration |
| `--batch-size <n>` | Override batch size |
| `--inference-steps <n>` | Override diffusion steps |

### 2. `benchmark` — Configuration Matrix

Runs a matrix of configurations and produces a summary table. Automatically adapts to GPU memory limits.

```bash
python profile_inference.py --mode benchmark
```

**Default matrix:**
- Durations: 30s, 60s, 120s, 240s (clamped by GPU memory)
- Batch sizes: 1, 2, 4 (clamped by GPU memory)
- Thinking: True, False
- Inference steps: 8, 16

**Output example:**

```
Duration   Batch   Think   Steps   Wall(s)    LM(s)      DiT(s)     VAE(s)     Status
--------------------------------------------------------------------------------------------------------------------------
30         1       False   8       3.21       0.45       1.89       0.52       OK
30         1       True    8       5.67       2.91       1.89       0.52       OK
60         2       False   16      12.34      0.48       9.12       1.85       OK
...
```

**Save results to JSON:**

```bash
python profile_inference.py --mode benchmark --benchmark-output results.json
```

### 3. `understand` — Audio Understanding Profiling

Profiles the `understand_music()` API which extracts metadata (BPM, key, time signature, caption) from audio codes.

```bash
python profile_inference.py --mode understand
python profile_inference.py --mode understand --audio-codes "your_audio_codes_string"
```

### 4. `create_sample` — Inspiration Mode Profiling

Profiles the `create_sample()` API which generates a complete song blueprint from a simple text query.

```bash
python profile_inference.py --mode create_sample
python profile_inference.py --mode create_sample --sample-query "a soft Bengali love song"
python profile_inference.py --mode create_sample --instrumental
```

### 5. `format_sample` — Metadata Formatting Profiling

Profiles the `format_sample()` API which converts caption + lyrics into structured metadata.

```bash
python profile_inference.py --mode format_sample
```

### 6. `tier-test` — Automated GPU Tier Testing

Automatically simulates different GPU VRAM sizes using `MAX_CUDA_VRAM` and runs a generation test at each tier. This is the recommended way to validate that all GPU tiers work correctly after modifying `acestep/gpu_config.py`.

```bash
# Test all tiers (4, 6, 8, 12, 16, 20, 24 GB)
python profile_inference.py --mode tier-test

# Test specific VRAM sizes
python profile_inference.py --mode tier-test --tiers 6 8 16

# Test with LM enabled (where the tier supports it)
python profile_inference.py --mode tier-test --tier-with-lm

# Quick test: skip torch.compile for non-quantized tiers
python profile_inference.py --mode tier-test --tier-skip-compile
```

**What it validates per tier:**
- Correct tier detection and `GPUConfig` construction
- Model initialization (DiT, VAE, Text Encoder, optionally LM)
- A short generation run (30s duration, batch=1) completes without OOM
- Adaptive VAE decode fallback (GPU → CPU offload → full CPU)
- VRAM usage stays within the simulated limit

**Output example:**

```
TIER TEST RESULTS
====================================================================================================
  VRAM    Tier       LM      Duration   Status    Peak VRAM    Notes
  ──────────────────────────────────────────────────────────────────────────────
  4GB     tier1      —       30s        ✅ OK     3.8GB        VAE decoded on CPU
  6GB     tier2      —       30s        ✅ OK     5.4GB        Tiled VAE chunk=256
  8GB     tier4      0.6B    30s        ✅ OK     7.2GB        vllm backend
  12GB    tier5      1.7B    30s        ✅ OK     10.8GB       vllm backend
  16GB    tier6a     1.7B    30s        ✅ OK     14.5GB       offload enabled
  20GB    tier6b     1.7B    30s        ✅ OK     17.2GB       no offload
  24GB    unlimited  4B      30s        ✅ OK     21.3GB       full models on GPU
```

> **Note**: `tier-test` mode uses `torch.cuda.set_per_process_memory_fraction()` to enforce a hard VRAM cap, making simulations realistic even on high-end GPUs (e.g., A100 80GB).

#### Boundary Testing

Use `--tier-boundary` to find the minimum VRAM tier at which INT8 quantization and CPU offload can be safely disabled. For each tier, up to three configurations are tested:

1. **default** — tier's standard settings
2. **no-quant** — quantization disabled, offload unchanged
3. **no-offload** — no quantization AND no CPU offload

```bash
# Run boundary tests across all tiers
python profile_inference.py --mode tier-test --tier-boundary

# Boundary test with LM enabled
python profile_inference.py --mode tier-test --tier-boundary --tier-with-lm

# Save boundary results to JSON
python profile_inference.py --mode tier-test --tier-boundary --benchmark-output boundary_results.json
```

The output includes a **Boundary Analysis** summary showing the minimum tier for each capability.

#### Batch Size Boundary Testing

Use `--tier-batch-boundary` to find the maximum safe batch size for each tier. For each tier, the tool progressively tests batch sizes 1, 2, 4, 8 (stopping at first OOM) with both LM-enabled and LM-disabled configurations:

```bash
# Run batch boundary tests
python profile_inference.py --mode tier-test --tier-batch-boundary --tier-with-lm

# Test specific tiers
python profile_inference.py --mode tier-test --tier-batch-boundary --tier-with-lm --tiers 8 12 16 24
```

The output includes a **Batch Boundary Summary** showing the maximum successful batch size per tier for both with-LM and without-LM configurations.

---

## CLI Reference

### Device & Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | Device: `auto` / `cuda` / `mps` / `cpu` |
| `--lm-backend` | `auto` | LLM backend: `auto` / `vllm` / `pt` / `mlx` |

### Model Paths

| Flag | Default | Description |
|------|---------|-------------|
| `--config-path` | `acestep-v15-turbo` | DiT model config |
| `--lm-model` | `acestep-5Hz-lm-1.7B` | LLM model path |

### Hardware Options

| Flag | Default | Description |
|------|---------|-------------|
| `--offload-to-cpu` | off | Offload models to CPU when not in use |
| `--offload-dit-to-cpu` | off | Offload DiT to CPU when not in use |
| `--quantization` | none | Quantization: `int8_weight_only` / `fp8_weight_only` / `w8a8_dynamic` |

### Generation Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | from example | Audio duration in seconds |
| `--batch-size` | from example | Batch size |
| `--inference-steps` | from example | Diffusion inference steps |
| `--seed` | from example | Random seed |
| `--guidance-scale` | 7.0 | CFG guidance scale for DiT |

### LLM / CoT Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--thinking` | off | Enable LLM Chain-of-Thought reasoning |
| `--use-cot-metas` | off | LLM generates music metadata via CoT |
| `--use-cot-caption` | off | LLM rewrites/formats caption via CoT |
| `--use-cot-language` | off | LLM detects vocal language via CoT |
| `--use-constrained-decoding` | on | FSM-based constrained decoding |
| `--no-constrained-decoding` | — | Disable constrained decoding |
| `--lm-temperature` | 0.85 | LLM sampling temperature |
| `--lm-cfg-scale` | 2.0 | LLM CFG scale |

### Profiling Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `profile` | Mode: `profile` / `benchmark` / `tier-test` / `understand` / `create_sample` / `format_sample` |
| `--no-warmup` | off | Skip warmup run |
| `--detailed` | off | Enable `cProfile` function-level analysis |
| `--llm-debug` | off | Deep LLM debugging (token count, throughput) |
| `--benchmark-output` | none | Save benchmark results to JSON file |

### Tier-Test Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tiers` | `4 6 8 12 16 20 24` | VRAM sizes (GB) to simulate |
| `--tier-with-lm` | off | Enable LM initialization on tiers that support it |
| `--tier-skip-compile` | off | Skip `torch.compile` for faster iteration on non-quantized tiers |
| `--tier-boundary` | off | Test each tier with no-quant and no-offload variants to find minimum capability boundaries |
| `--tier-batch-boundary` | off | Test each tier with batch sizes 1, 2, 4, 8 to find maximum safe batch size |

### Input Options

| Flag | Default | Description |
|------|---------|-------------|
| `--example` | `example_05.json` | Example JSON from `examples/text2music/` |
| `--task-type` | `text2music` | Task: `text2music` / `cover` / `repaint` / `lego` / `extract` / `complete` |
| `--reference-audio` | none | Reference audio path (for cover/style transfer) |
| `--src-audio` | none | Source audio path (for audio-to-audio tasks) |
| `--sample-query` | none | Query for `create_sample` mode |
| `--instrumental` | off | Generate instrumental music (for `create_sample`) |
| `--audio-codes` | none | Audio codes string (for `understand` mode) |

---

## Examples

### Compare Devices

```bash
# NVIDIA GPU
python profile_inference.py --device cuda --lm-backend vllm

# Apple Silicon
python profile_inference.py --device mps --lm-backend mlx

# CPU baseline
python profile_inference.py --device cpu --lm-backend pt
```

### Compare LLM Models

```bash
# Lightweight (0.6B)
python profile_inference.py --lm-model acestep-5Hz-lm-0.6B

# Default (1.7B)
python profile_inference.py --lm-model acestep-5Hz-lm-1.7B

# Large (4B)
python profile_inference.py --lm-model acestep-5Hz-lm-4B
```

### Thinking vs No-Thinking

```bash
# Without thinking (faster)
python profile_inference.py --mode benchmark

# With thinking (better quality, slower)
python profile_inference.py --thinking --use-cot-metas --use-cot-caption
```

### Low-VRAM Profiling

```bash
# Offload + quantization
python profile_inference.py --offload-to-cpu --quantization int8_weight_only --lm-model acestep-5Hz-lm-0.6B
```

### Full Benchmark Suite

```bash
# Run full benchmark matrix and save results
python profile_inference.py --mode benchmark --benchmark-output benchmark_results.json

# Then inspect the JSON
cat benchmark_results.json | python -m json.tool
```

### Function-Level Profiling

```bash
# Enable cProfile for detailed function-level analysis
python profile_inference.py --detailed --llm-debug
```

---

## Understanding Output

### Time Costs Breakdown

The profiler prints a detailed breakdown of where time is spent:

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

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Wall Time** | End-to-end time from start to finish |
| **LM Total Time** | Time spent in LLM planning (token generation + parsing) |
| **DiT Total Time** | Time spent in diffusion (all steps combined) |
| **VAE Decode Time** | Time to decode latents to audio waveform |
| **Tokens/sec** | LLM token generation throughput (with `--llm-debug`) |

---

## Tips & Best Practices

1. **Always include warmup** (default) — The first run includes JIT compilation and memory allocation overhead. Warmup ensures measurements reflect steady-state performance.

2. **Use `--benchmark-output`** to save results as JSON for later analysis or comparison across hardware.

3. **Compare with thinking off vs on** — Thinking mode significantly increases LLM time but may improve generation quality.

4. **Test with representative durations** — Short durations (30s) are dominated by LLM time; long durations (240s+) are dominated by DiT time.

5. **GPU memory auto-adaptation** — The benchmark mode automatically clamps durations and batch sizes to what your GPU can handle, using the adaptive tier system in `acestep/gpu_config.py`.

6. **Use `--detailed` sparingly** — `cProfile` adds overhead; use it only when investigating function-level bottlenecks.

7. **Use `tier-test` for regression testing** — After modifying GPU tier configs, run `--mode tier-test` to verify all tiers still work correctly. This is especially important when changing offload thresholds, duration limits, or LM model availability.

8. **Simulate low VRAM realistically** — When using `MAX_CUDA_VRAM`, the system enforces a hard VRAM cap via `set_per_process_memory_fraction()`, so OOM errors during simulation reflect real behavior on consumer GPUs.
