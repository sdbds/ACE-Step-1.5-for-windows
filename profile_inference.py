#!/usr/bin/env python3
"""
ACE-Step 1.5 Inference Profiler & Benchmark

Comprehensive profiling tool that supports all features, devices, and backends.
Uses the high-level inference API and built-in time_costs for accurate timing.

Modes:
    profile         - Profile a single generation run with detailed timing breakdown
    benchmark       - Run a matrix of configurations and produce a summary table
    understand      - Profile the understand_music() API (audio codes -> metadata)
    create_sample   - Profile the create_sample() API (inspiration/simple mode)
    format_sample   - Profile the format_sample() API (caption+lyrics -> metadata)

Usage:
    # Profile text2music with default settings
    python profile_inference.py

    # Profile with thinking enabled on MPS
    python profile_inference.py --device mps --thinking

    # Benchmark across configurations
    python profile_inference.py --mode benchmark

    # Profile create_sample (inspiration mode)
    python profile_inference.py --mode create_sample --sample-query "a soft Bengali love song"

    # Profile understand mode
    python profile_inference.py --mode understand

    # Full profiling with cProfile
    python profile_inference.py --detailed --llm-debug
"""

import time
import argparse
import sys
import os
import json
import tempfile
from contextlib import contextmanager
from collections import defaultdict
from typing import Tuple, Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from acestep.inference import (
    generate_music,
    understand_music,
    create_sample,
    format_sample,
    GenerationParams,
    GenerationConfig,
    GenerationResult,
)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.gpu_config import get_gpu_config, set_global_gpu_config


# =============================================================================
# Device / Backend helpers
# =============================================================================


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to the best available device."""
    if device == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def auto_detect_backend(device: str) -> str:
    """Auto-detect the best LLM backend for the resolved device."""
    if device == "mps":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            return "pt"
    if device.startswith("cuda"):
        return "vllm"
    return "pt"


def load_env_config() -> Dict[str, str]:
    """Load configuration defaults from .env file."""
    env_config = {
        "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
        "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
        "ACESTEP_DEVICE": "auto",
        "ACESTEP_LM_BACKEND": "auto",
    }
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in env_config and value:
                        env_config[key] = value
    return env_config


# =============================================================================
# Timer utilities
# =============================================================================


class PreciseTimer:
    """High-precision timer with GPU synchronization for accurate timing."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.enabled = True
        
    def sync(self):
        """Synchronize GPU operations for accurate timing."""
        if not self.enabled:
            return
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif self.device.startswith("xpu") and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
    
    @contextmanager
    def time(self, name: str):
        """Time a code section with GPU synchronization."""
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)
    
    def get_total(self, name: str) -> float:
        return sum(self.timings.get(name, []))
    
    def get_mean(self, name: str) -> float:
        times = self.timings.get(name, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_count(self, name: str) -> int:
        return len(self.timings.get(name, []))
    
    def reset(self):
        self.timings.clear()


# =============================================================================
# Example config loader
# =============================================================================


def load_example_config(
    example_file: str, cli_overrides: argparse.Namespace
) -> Tuple[Optional[GenerationParams], Optional[GenerationConfig]]:
    """Load configuration from example JSON file, applying CLI overrides."""
    try:
        with open(example_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = GenerationParams(
            caption=data.get("caption", ""),
            lyrics=data.get("lyrics", ""),
            bpm=data.get("bpm"),
            keyscale=data.get("keyscale", ""),
            timesignature=data.get("timesignature", ""),
            vocal_language=data.get("language", "unknown"),
            duration=(
                cli_overrides.duration
                if cli_overrides.duration is not None
                else data.get("duration", -1.0)
            ),
            thinking=cli_overrides.thinking,
            use_cot_metas=cli_overrides.use_cot_metas,
            use_cot_caption=cli_overrides.use_cot_caption,
            use_cot_language=cli_overrides.use_cot_language,
            use_constrained_decoding=cli_overrides.use_constrained_decoding,
            inference_steps=(
                cli_overrides.inference_steps
                if cli_overrides.inference_steps is not None
                else data.get("inference_steps", 8)
            ),
            seed=(
                cli_overrides.seed
                if cli_overrides.seed is not None
                else data.get("seed", 42)
            ),
            task_type=cli_overrides.task_type,
            lm_temperature=cli_overrides.lm_temperature,
            lm_cfg_scale=cli_overrides.lm_cfg_scale,
            guidance_scale=cli_overrides.guidance_scale,
            reference_audio=cli_overrides.reference_audio,
            src_audio=cli_overrides.src_audio,
        )

        config = GenerationConfig(
            batch_size=(
                cli_overrides.batch_size
                if cli_overrides.batch_size is not None
                else data.get("batch_size", 1)
            ),
            seeds=[params.seed] if params.seed >= 0 else None,
            use_random_seed=(params.seed < 0),
            audio_format="flac",
        )

        return params, config

    except Exception as e:
        print(f"  Failed to load example: {e}")
        return None, None


# =============================================================================
# Printing helpers
# =============================================================================


def print_time_costs_breakdown(
    time_costs: Dict[str, float], total_wall_time: float
):
    """Print a detailed timing breakdown from result.extra_outputs['time_costs']."""
        print("\n" + "=" * 100)
    print("PROFILING RESULTS")
        print("=" * 100)
        
    if not time_costs:
        print("\n  (No time_costs data available from the pipeline)")
        print(f"\n  Total wall time: {total_wall_time:.3f}s")
        return

    # Categorize keys
    lm_keys = {
        k: v
        for k, v in time_costs.items()
        if k.startswith("lm_") and isinstance(v, (int, float))
    }
    dit_keys = {
        k: v
        for k, v in time_costs.items()
        if k.startswith("dit_") and isinstance(v, (int, float))
    }
    pipeline_keys = {
        k: v
        for k, v in time_costs.items()
        if k.startswith("pipeline_") and isinstance(v, (int, float))
    }
    other_keys = {
        k: v
        for k, v in time_costs.items()
        if not k.startswith(("lm_", "dit_", "pipeline_"))
        and isinstance(v, (int, float))
    }

    print(f"\n{'COMPONENT':<50} {'TIME (s)':<12} {'% of wall':<10}")
    print("-" * 72)

    # LM timing
    lm_total = lm_keys.get("lm_total_time", 0.0)
    if lm_keys:
        print(
            f"\n{'LLM (5Hz Language Model)':<50} "
            f"{lm_total:<12.3f} {100 * lm_total / total_wall_time:>6.1f}%"
        )
        for k, v in sorted(lm_keys.items()):
            if k != "lm_total_time":
                label = k.replace("lm_", "  ")
                print(
                    f"  {label:<48} "
                    f"{v:<12.3f} {100 * v / total_wall_time:>6.1f}%"
                )

    # DiT timing
    dit_total = dit_keys.get("dit_total_time_cost", 0.0)
    if dit_keys:
        print(
            f"\n{'DiT (Diffusion Transformer)':<50} "
            f"{dit_total:<12.3f} {100 * dit_total / total_wall_time:>6.1f}%"
        )
        for k, v in sorted(dit_keys.items()):
            if k != "dit_total_time_cost":
                label = k.replace("dit_", "  ")
                print(
                    f"  {label:<48} "
                    f"{v:<12.3f} {100 * v / total_wall_time:>6.1f}%"
                )

    # Pipeline total
    if pipeline_keys:
        for k, v in sorted(pipeline_keys.items()):
            print(
                f"\n{'Pipeline: ' + k:<50} "
                f"{v:<12.3f} {100 * v / total_wall_time:>6.1f}%"
            )

    # Other keys
    if other_keys:
        print(f"\n{'Other:':<50}")
        for k, v in sorted(other_keys.items()):
            print(
                f"  {k:<48} "
                f"{v:<12.3f} {100 * v / total_wall_time:>6.1f}%"
            )

    # Overhead (wall time minus accounted time)
    accounted = lm_total + dit_total
    overhead = total_wall_time - accounted
    if overhead > 0.01:
        print(
            f"\n{'Overhead (I/O, audio save, etc.)':<50} "
            f"{overhead:<12.3f} {100 * overhead / total_wall_time:>6.1f}%"
        )

    print(f"\n{'TOTAL WALL TIME':<50} {total_wall_time:<12.3f} {'100.0%':>6}")

    # Performance insights
        print("\n" + "=" * 100)
    print("PERFORMANCE INSIGHTS")
        print("=" * 100)
        
    if lm_total > 0 and dit_total > 0:
        if lm_total > dit_total * 2:
            print(
                f"\n  LLM is the bottleneck: {lm_total:.1f}s "
                f"({100 * lm_total / total_wall_time:.0f}% of total)"
            )
            print("  Suggestions:")
            print("    1. Run with --llm-debug for token-level throughput analysis")
            print("    2. Try --no-constrained-decoding to reduce FSM overhead")
            print("    3. Compare backends: --lm-backend vllm vs pt vs mlx")
            print(
                "    4. Reduce lm_cfg_scale "
                "(currently doubles forward passes if > 1.0)"
            )
        elif dit_total > lm_total * 2:
            print(
                f"\n  DiT is the bottleneck: {dit_total:.1f}s "
                f"({100 * dit_total / total_wall_time:.0f}% of total)"
            )
            print("  Suggestions:")
            print("    1. Reduce --inference-steps (turbo model supports 4-8)")
            print("    2. Reduce --duration")
            print("    3. Try --quantization int8_weight_only")
        else:
            print(
                f"\n  Balanced pipeline: LLM={lm_total:.1f}s, DiT={dit_total:.1f}s"
            )
    elif dit_total > 0:
        print(f"\n  DiT only (no LLM): {dit_total:.1f}s")
        vae_time = dit_keys.get("dit_vae_decode_time_cost", 0.0)
        diffusion_time = dit_keys.get(
            "dit_diffusion_time_cost", dit_total - vae_time
        )
        if vae_time > 0:
            print(
                f"    Diffusion: {diffusion_time:.1f}s, "
                f"VAE decode: {vae_time:.1f}s"
            )


def print_result_summary(result: GenerationResult, mode: str = "profile"):
    """Print a short summary of the generation result."""
    if result.success:
        n_audios = len(result.audios)
        silent_count = sum(1 for a in result.audios if a.get("silent", False))
        print(f"\n  Success! Generated {n_audios} audio(s)", end="")
        if silent_count:
            print(f" ({silent_count} silent)", end="")
        print()
        else:
        print(f"\n  FAILED: {result.error}")


# =============================================================================
# Mode: profile (text2music and other task types)
# =============================================================================


def run_profile_mode(dit_handler, llm_handler, args, timer: PreciseTimer):
    """Run a single profiled generation."""
    example_dir = "text2music"
    example_file = os.path.join(
        PROJECT_ROOT, "examples", example_dir, args.example
    )
    if not os.path.exists(example_file):
        print(f"\n  Example not found: {example_file}")
        sys.exit(1)

    print(f"\n  Loading example: {args.example}")
    params, config = load_example_config(example_file, args)
    if not params or not config:
        print("  Failed to load example config")
        sys.exit(1)

    caption_preview = (
        params.caption[:80] + "..."
        if len(params.caption) > 80
        else params.caption
    )
    print(f"  Caption: {caption_preview}")
    print(
        f"  Task: {params.task_type}, Batch: {config.batch_size}, "
        f"Steps: {params.inference_steps}"
    )
    print(
        f"  Thinking: {params.thinking}, CoT Metas: {params.use_cot_metas}, "
        f"CoT Caption: {params.use_cot_caption}"
    )

    # Use a temporary directory for output (don't pollute project root)
    save_dir = tempfile.mkdtemp(prefix="acestep_profile_")

    # Warmup
    if not args.no_warmup:
        print("\n" + "-" * 100)
        print("WARMUP RUN")
        print("-" * 100)
        warmup_params = GenerationParams(
            caption=params.caption,
            lyrics=params.lyrics,
            bpm=params.bpm,
            keyscale=params.keyscale,
            timesignature=params.timesignature,
            vocal_language=params.vocal_language,
            duration=params.duration,
            thinking=params.thinking,
            use_cot_metas=params.use_cot_metas,
            use_cot_caption=params.use_cot_caption,
            use_cot_language=params.use_cot_language,
            use_constrained_decoding=params.use_constrained_decoding,
            inference_steps=params.inference_steps,
            seed=42,
            task_type=params.task_type,
            lm_temperature=params.lm_temperature,
            lm_cfg_scale=params.lm_cfg_scale,
            guidance_scale=params.guidance_scale,
        )
        warmup_config = GenerationConfig(
            batch_size=1, seeds=[42], use_random_seed=False, audio_format="flac"
        )
        warmup_start = time.perf_counter()
        warmup_result = generate_music(
            dit_handler, llm_handler, warmup_params, warmup_config,
            save_dir=save_dir,
        )
        warmup_time = time.perf_counter() - warmup_start
        print(f"  Warmup completed: {warmup_time:.2f}s")
        if not warmup_result.success:
            print(f"  Warning: warmup failed: {warmup_result.error}")
        timer.reset()

    # Profiling run
    print("\n" + "=" * 100)
    print("PROFILING RUN")
    print("=" * 100)

    # Optional cProfile
    prof = None
    if args.detailed:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()

    timer.sync()
    total_start = time.perf_counter()

    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=save_dir
    )

    timer.sync()
    total_wall_time = time.perf_counter() - total_start

    if args.detailed and prof:
        prof.disable()
        _print_cprofile(prof)

    # Print results
    print_result_summary(result, "profile")

    time_costs = (
        result.extra_outputs.get("time_costs", {}) if result.success else {}
    )
    print_time_costs_breakdown(time_costs, total_wall_time)

    # Cleanup temp dir
    _cleanup_dir(save_dir)

    return result, total_wall_time


# =============================================================================
# Mode: benchmark
# =============================================================================


def run_benchmark_mode(dit_handler, llm_handler, args, timer: PreciseTimer):
    """Run a matrix of configurations and produce a summary table."""
    example_file = os.path.join(
        PROJECT_ROOT, "examples", "text2music", args.example
    )
    if not os.path.exists(example_file):
        print(f"\n  Example not found: {example_file}")
        sys.exit(1)

    with open(example_file, "r", encoding="utf-8") as f:
        example_data = json.load(f)

    save_dir = tempfile.mkdtemp(prefix="acestep_bench_")

    # Define benchmark matrix
    durations = [30, 60, 120]
    batch_sizes = [1, 2]
    thinking_options = (
        [False, True] if llm_handler.llm_initialized else [False]
    )
    inference_steps_options = [8]

    # Clamp to GPU limits
    gpu_config = get_gpu_config()
    max_dur = gpu_config.max_duration_without_lm
    max_batch = gpu_config.max_batch_size_without_lm
    durations = [d for d in durations if d <= max_dur]
    batch_sizes = [b for b in batch_sizes if b <= max_batch]

    if not durations:
        durations = [30]
    if not batch_sizes:
        batch_sizes = [1]

    configs = []
    for dur in durations:
        for bs in batch_sizes:
            for think in thinking_options:
                for steps in inference_steps_options:
                    configs.append(
                        {
                            "duration": dur,
                            "batch_size": bs,
                            "thinking": think,
                            "inference_steps": steps,
                        }
                    )

    print(f"\n  Running {len(configs)} benchmark configurations...")
    print(f"  Durations: {durations}, Batch sizes: {batch_sizes}")
    print(f"  Thinking: {thinking_options}, Steps: {inference_steps_options}")

    # Warmup
    if not args.no_warmup:
        print("\n  Warmup run...")
        warmup_params = GenerationParams(
            caption=example_data.get("caption", ""),
            lyrics=example_data.get("lyrics", ""),
            duration=30,
            thinking=False,
            inference_steps=8,
            seed=42,
        )
        warmup_config = GenerationConfig(
            batch_size=1, seeds=[42], use_random_seed=False, audio_format="flac"
        )
        generate_music(
            dit_handler, llm_handler, warmup_params, warmup_config,
            save_dir=save_dir,
        )
        print("  Warmup done.")

    # Run benchmark
    results = []
    for i, cfg in enumerate(configs):
        label = (
            f"dur={cfg['duration']}s, bs={cfg['batch_size']}, "
            f"think={cfg['thinking']}, steps={cfg['inference_steps']}"
        )
        print(f"\n  [{i + 1}/{len(configs)}] {label}")

        params = GenerationParams(
            caption=example_data.get("caption", ""),
            lyrics=example_data.get("lyrics", ""),
            bpm=example_data.get("bpm"),
            keyscale=example_data.get("keyscale", ""),
            timesignature=example_data.get("timesignature", ""),
            vocal_language=example_data.get("language", "unknown"),
            duration=cfg["duration"],
            thinking=cfg["thinking"],
            use_cot_metas=cfg["thinking"],
            use_cot_caption=cfg["thinking"],
            use_cot_language=cfg["thinking"],
            use_constrained_decoding=args.use_constrained_decoding,
            inference_steps=cfg["inference_steps"],
            seed=42,
            lm_temperature=args.lm_temperature,
            lm_cfg_scale=args.lm_cfg_scale,
            guidance_scale=args.guidance_scale,
        )
        config = GenerationConfig(
            batch_size=cfg["batch_size"],
            seeds=[42 + j for j in range(cfg["batch_size"])],
            use_random_seed=False,
            audio_format="flac",
        )

        timer.sync()
        t0 = time.perf_counter()
        result = generate_music(
            dit_handler, llm_handler, params, config, save_dir=save_dir
        )
        timer.sync()
        wall_time = time.perf_counter() - t0

        tc = (
            result.extra_outputs.get("time_costs", {})
            if result.success
            else {}
        )
        entry = {
            "config": cfg,
            "wall_time": wall_time,
            "success": result.success,
            "error": result.error,
            "lm_time": tc.get("lm_total_time", 0.0),
            "dit_time": tc.get("dit_total_time_cost", 0.0),
            "vae_time": tc.get("dit_vae_decode_time_cost", 0.0),
            "n_audios": len(result.audios) if result.success else 0,
        }
        results.append(entry)

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(
            f"    {status} | wall={wall_time:.1f}s, "
            f"lm={entry['lm_time']:.1f}s, dit={entry['dit_time']:.1f}s"
        )
    
    # Print summary table
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    header = (
        f"{'Duration':<10} {'Batch':<7} {'Think':<7} {'Steps':<7} "
        f"{'Wall(s)':<10} {'LM(s)':<10} {'DiT(s)':<10} "
        f"{'VAE(s)':<10} {'Status':<10}"
    )
    print(header)
    print("-" * 120)

    for entry in results:
        cfg = entry["config"]
        status = "OK" if entry["success"] else "FAIL"
        print(
            f"{cfg['duration']:<10} {cfg['batch_size']:<7} "
            f"{str(cfg['thinking']):<7} {cfg['inference_steps']:<7} "
            f"{entry['wall_time']:<10.2f} {entry['lm_time']:<10.2f} "
            f"{entry['dit_time']:<10.2f} {entry['vae_time']:<10.2f} "
            f"{status:<10}"
        )

    # Save benchmark results as JSON
    if args.benchmark_output:
        output_path = args.benchmark_output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Benchmark results saved to: {output_path}")

    _cleanup_dir(save_dir)
    return results


# =============================================================================
# Mode: understand
# =============================================================================


def run_understand_mode(dit_handler, llm_handler, args, timer: PreciseTimer):
    """Profile the understand_music() API."""
    if not llm_handler.llm_initialized:
        print("\n  LLM not initialized. understand mode requires LLM.")
        print("  Re-run with --thinking or ensure LLM is available.")
        sys.exit(1)

    audio_codes = args.audio_codes if args.audio_codes else ""

    print(
        f"\n  Audio codes: "
        f"{'<provided>' if audio_codes else '<empty - will generate sample>'}"
    )

    timer.sync()
    t0 = time.perf_counter()

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=audio_codes,
        temperature=args.lm_temperature,
        use_constrained_decoding=args.use_constrained_decoding,
    )

    timer.sync()
    wall_time = time.perf_counter() - t0

    print(f"\n  Wall time: {wall_time:.3f}s")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Caption: {result.caption[:100]}...")
        print(
            f"  BPM: {result.bpm}, Duration: {result.duration}, "
            f"Key: {result.keyscale}"
        )
        print(
            f"  Language: {result.language}, Time Sig: {result.timesignature}"
        )
        if result.lyrics:
            print(f"  Lyrics: {result.lyrics[:100]}...")
    else:
        print(f"  Error: {result.error}")

    return result, wall_time


# =============================================================================
# Mode: create_sample
# =============================================================================


def run_create_sample_mode(
    dit_handler, llm_handler, args, timer: PreciseTimer
):
    """Profile the create_sample() API (inspiration/simple mode)."""
    if not llm_handler.llm_initialized:
        print("\n  LLM not initialized. create_sample mode requires LLM.")
        sys.exit(1)

    query = args.sample_query or "a soft love song for a quiet evening"
    print(f"\n  Query: {query}")
    print(f"  Instrumental: {args.instrumental}")

        timer.sync()
    t0 = time.perf_counter()

    result = create_sample(
        llm_handler=llm_handler,
        query=query,
        instrumental=args.instrumental,
        temperature=args.lm_temperature,
        use_constrained_decoding=args.use_constrained_decoding,
    )

    timer.sync()
    wall_time = time.perf_counter() - t0

    print(f"\n  Wall time: {wall_time:.3f}s")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Caption: {result.caption[:100]}...")
        print(
            f"  BPM: {result.bpm}, Duration: {result.duration}, "
            f"Key: {result.keyscale}"
        )
        print(
            f"  Language: {result.language}, Time Sig: {result.timesignature}"
        )
        print(f"  Instrumental: {result.instrumental}")
        if result.lyrics:
            print(f"  Lyrics: {result.lyrics[:100]}...")
    else:
        print(f"  Error: {result.error}")

    return result, wall_time


# =============================================================================
# Mode: format_sample
# =============================================================================


def run_format_sample_mode(
    dit_handler, llm_handler, args, timer: PreciseTimer
):
    """Profile the format_sample() API."""
    if not llm_handler.llm_initialized:
        print("\n  LLM not initialized. format_sample mode requires LLM.")
        sys.exit(1)

    example_file = os.path.join(
        PROJECT_ROOT, "examples", "text2music", args.example
    )
    if not os.path.exists(example_file):
        print(f"\n  Example not found: {example_file}")
        sys.exit(1)

    with open(example_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    caption = data.get("caption", "Latin pop, reggaeton")
    lyrics = data.get("lyrics", "[Verse 1]\nHola mundo")

    print(f"\n  Caption: {caption[:80]}...")
    print(f"  Lyrics: {lyrics[:80]}...")

        timer.sync()
    t0 = time.perf_counter()

    result = format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        temperature=args.lm_temperature,
        use_constrained_decoding=args.use_constrained_decoding,
    )

    timer.sync()
    wall_time = time.perf_counter() - t0

    print(f"\n  Wall time: {wall_time:.3f}s")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Caption: {result.caption[:100]}...")
        print(
            f"  BPM: {result.bpm}, Duration: {result.duration}, "
            f"Key: {result.keyscale}"
        )
        print(
            f"  Language: {result.language}, Time Sig: {result.timesignature}"
        )
    else:
        print(f"  Error: {result.error}")

    return result, wall_time


# =============================================================================
# cProfile helper
# =============================================================================


def _print_cprofile(prof):
    """Print cProfile results and save to file."""
            import pstats
            import io
            
            output_file = "profile_cprofile_detailed.txt"
    with open(output_file, "w") as f:
                ps = pstats.Stats(prof, stream=f)
        ps.sort_stats("cumulative")
                ps.print_stats(100)
            
            print("\n" + "=" * 100)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME (cProfile)")
            print("=" * 100)
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s)
    ps.sort_stats("cumulative")
            ps.print_stats(20)
            print(s.getvalue())
    print(f"Full report saved to: {output_file}")


def _cleanup_dir(path: str):
    """Remove temporary directory silently."""
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# =============================================================================
# Handler initialization
# =============================================================================


def initialize_handlers(
    args, device: str
) -> Tuple[AceStepHandler, LLMHandler]:
    """Initialize DiT and LLM handlers with current API."""
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # Determine flash attention availability
    use_flash_attention = False
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            use_flash_attention = True
        except ImportError:
            pass

    print("  Initializing DiT handler...")
    status_dit, success_dit = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=args.config_path,
        device=args.device,  # Pass original device string (handler resolves "auto")
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=args.offload_dit_to_cpu,
        quantization=args.quantization,
    )
    if not success_dit:
        print(f"  DiT initialization failed: {status_dit}")
        sys.exit(1)
    print(f"  DiT ready (device={dit_handler.device})")

    # Determine if LLM should be initialized
    need_llm = (
        args.thinking
        or args.use_cot_metas
        or args.use_cot_caption
        or args.use_cot_language
        or args.mode in ("understand", "create_sample", "format_sample")
    )

    if need_llm:
        print(f"  Initializing LLM handler (backend={args.lm_backend})...")
        status_llm, success_llm = llm_handler.initialize(
            checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
            lm_model_path=args.lm_model,
            backend=args.lm_backend,
            device=args.device,
            offload_to_cpu=args.offload_to_cpu,
            dtype=None,
        )
        if success_llm:
            print(f"  LLM ready (backend={llm_handler.llm_backend})")
        else:
            print(f"  LLM initialization failed: {status_llm}")
            if args.mode in ("understand", "create_sample", "format_sample"):
                sys.exit(1)
    else:
        print(
            "  LLM not needed for current configuration "
            "(thinking/CoT disabled)"
        )

    return dit_handler, llm_handler


# =============================================================================
# CLI argument parser
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all options."""
    env_config = load_env_config()
    
    parser = argparse.ArgumentParser(
        description="ACE-Step 1.5 Inference Profiler & Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_inference.py                                    # Profile text2music
  python profile_inference.py --thinking --llm-debug             # With LLM analysis
  python profile_inference.py --mode benchmark                   # Benchmark matrix
  python profile_inference.py --mode understand                  # Profile understand API
  python profile_inference.py --mode create_sample --sample-query "jazz ballad"
  python profile_inference.py --device mps --lm-backend mlx      # Apple Silicon
  python profile_inference.py --device cuda --lm-backend vllm    # NVIDIA GPU
""",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=[
            "profile",
            "benchmark",
            "understand",
            "create_sample",
            "format_sample",
        ],
        help="Profiling mode (default: profile)",
    )

    # Device & backend
    parser.add_argument(
        "--device",
        type=str,
        default=env_config["ACESTEP_DEVICE"],
        help=(
            f"Device: auto/cuda/mps/cpu "
            f"(default: {env_config['ACESTEP_DEVICE']})"
        ),
    )
    parser.add_argument(
        "--lm-backend",
        type=str,
        default=env_config["ACESTEP_LM_BACKEND"],
        choices=["auto", "vllm", "pt", "mlx"],
        help=(
            f"LLM backend "
            f"(default: {env_config['ACESTEP_LM_BACKEND']})"
        ),
    )

    # Model paths
    parser.add_argument(
        "--config-path",
        type=str,
        default=env_config["ACESTEP_CONFIG_PATH"],
        help=(
            f"DiT model config "
            f"(default: {env_config['ACESTEP_CONFIG_PATH']})"
        ),
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default=env_config["ACESTEP_LM_MODEL_PATH"],
        help=(
            f"LLM model path "
            f"(default: {env_config['ACESTEP_LM_MODEL_PATH']})"
        ),
    )

    # Hardware options
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Offload models to CPU when not in use",
    )
    parser.add_argument(
        "--offload-dit-to-cpu",
        action="store_true",
        help="Offload DiT to CPU when not in use",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int8_weight_only", "fp8_weight_only", "w8a8_dynamic"],
        help="Quantization mode for DiT model",
    )

    # Example & input
    parser.add_argument(
        "--example",
        type=str,
        default="example_05.json",
        help="Example JSON file from examples/text2music/",
    )

    # Task type
    parser.add_argument(
        "--task-type",
        type=str,
        default="text2music",
        choices=[
            "text2music",
            "cover",
            "repaint",
            "lego",
            "extract",
            "complete",
        ],
        help="Generation task type (default: text2music)",
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Reference audio path (for cover/style transfer)",
    )
    parser.add_argument(
        "--src-audio",
        type=str,
        default=None,
        help="Source audio path (for audio-to-audio tasks)",
    )

    # Generation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Audio duration in seconds (overrides example)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides example)",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Diffusion inference steps (overrides example)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides example)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="CFG guidance scale for DiT (default: 7.0)",
    )

    # LLM / CoT parameters
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable 5Hz LM Chain-of-Thought reasoning",
    )
    parser.add_argument(
        "--use-cot-metas",
        action="store_true",
        help="Enable LLM to generate music metadata via CoT",
    )
    parser.add_argument(
        "--use-cot-caption",
        action="store_true",
        help="Enable LLM to rewrite/format caption via CoT",
    )
    parser.add_argument(
        "--use-cot-language",
        action="store_true",
        help="Enable LLM to detect vocal language via CoT",
    )
    parser.add_argument(
        "--use-constrained-decoding",
        action="store_true",
        default=True,
        help="Use FSM-based constrained decoding (default: True)",
    )
    parser.add_argument(
        "--no-constrained-decoding",
        action="store_true",
        help="Disable constrained decoding",
    )
    parser.add_argument(
        "--lm-temperature",
        type=float,
        default=0.85,
        help="LLM sampling temperature (default: 0.85)",
    )
    parser.add_argument(
        "--lm-cfg-scale",
        type=float,
        default=2.0,
        help="LLM CFG scale (default: 2.0)",
    )

    # Profiling options
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run (includes compilation overhead)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Enable cProfile function-level analysis",
    )
    parser.add_argument(
        "--llm-debug",
        action="store_true",
        help="Enable deep LLM debugging (token count, throughput)",
    )

    # Benchmark options
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Save benchmark results to JSON file",
    )

    # create_sample / understand options
    parser.add_argument(
        "--sample-query",
        type=str,
        default=None,
        help="Query for create_sample mode",
    )
    parser.add_argument(
        "--instrumental",
        action="store_true",
        help="Generate instrumental music (for create_sample)",
    )
    parser.add_argument(
        "--audio-codes",
        type=str,
        default=None,
        help="Audio codes string (for understand mode)",
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle --no-constrained-decoding
    if args.no_constrained_decoding:
        args.use_constrained_decoding = False

    # Resolve device
    device = resolve_device(args.device)

    # Auto-detect backend
    if args.lm_backend == "auto":
        args.lm_backend = auto_detect_backend(device)

    # Setup GPU config
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)

    # Auto-enable offload for small GPUs
    if (
        gpu_config.gpu_memory_gb > 0
        and gpu_config.gpu_memory_gb < 16
        and not args.offload_to_cpu
    ):
        args.offload_to_cpu = True

    # Print header
    print("=" * 100)
    print("ACE-Step 1.5 Inference Profiler")
    print("=" * 100)
    print(f"\n  Mode:           {args.mode}")
    print(f"  Device:         {device} (requested: {args.device})")
    print(f"  LLM Backend:    {args.lm_backend}")
    print(f"  DiT Config:     {args.config_path}")
    print(f"  LLM Model:      {args.lm_model}")
    print(
        f"  GPU Memory:     {gpu_config.gpu_memory_gb:.1f} GB "
        f"(tier: {gpu_config.tier})"
    )
    if args.quantization:
        print(f"  Quantization:   {args.quantization}")
    if args.offload_to_cpu:
        print("  CPU Offload:    enabled")
    print(f"\n  Thinking:       {args.thinking}")
    print(f"  CoT Metas:      {args.use_cot_metas}")
    print(f"  CoT Caption:    {args.use_cot_caption}")
    print(f"  CoT Language:   {args.use_cot_language}")
    print(f"  Constrained:    {args.use_constrained_decoding}")
    print(f"  Warmup:         {'disabled' if args.no_warmup else 'enabled'}")

    # Initialize handlers
    print("\n" + "-" * 100)
    print("INITIALIZING MODELS")
    print("-" * 100)

    dit_handler, llm_handler = initialize_handlers(args, device)

    # Create timer with resolved device
    actual_device = getattr(dit_handler, "device", device)
    timer = PreciseTimer(device=actual_device)

    # Dispatch to mode
    print("\n" + "=" * 100)
    print(f"RUNNING MODE: {args.mode.upper()}")
    print("=" * 100)

    if args.mode == "profile":
        run_profile_mode(dit_handler, llm_handler, args, timer)
    elif args.mode == "benchmark":
        run_benchmark_mode(dit_handler, llm_handler, args, timer)
    elif args.mode == "understand":
        run_understand_mode(dit_handler, llm_handler, args, timer)
    elif args.mode == "create_sample":
        run_create_sample_mode(dit_handler, llm_handler, args, timer)
    elif args.mode == "format_sample":
        run_format_sample_mode(dit_handler, llm_handler, args, timer)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
