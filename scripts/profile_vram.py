#!/usr/bin/env python3
"""
VRAM Profiling Script for ACE-Step 1.5

Measures actual GPU memory consumption of each model component at different
configurations. Results are used to calibrate the empirical VRAM constants
in gpu_config.py.

Usage:
    python scripts/profile_vram.py                          # Profile all components
    python scripts/profile_vram.py --component dit          # Profile DiT only
    python scripts/profile_vram.py --component lm           # Profile LM only
    python scripts/profile_vram.py --component vae          # Profile VAE only
    python scripts/profile_vram.py --output results.json    # Save results to JSON

Requirements:
    - CUDA GPU with sufficient memory
    - All model checkpoints downloaded
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def get_memory_stats() -> Dict[str, float]:
    """Get current CUDA memory statistics in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0, "max_allocated": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    return {
        "allocated": round(allocated, 3),
        "reserved": round(reserved, 3),
        "free": round(free_gb, 3),
        "total": round(total_gb, 3),
        "max_allocated": round(max_allocated, 3),
    }


def reset_memory():
    """Reset CUDA memory stats and free caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        # Wait for GPU to settle
        torch.cuda.synchronize()


def measure_cuda_context() -> Dict[str, float]:
    """Measure CUDA context overhead."""
    print("\n" + "=" * 60)
    print("Measuring CUDA context overhead...")
    print("=" * 60)
    
    reset_memory()
    before = get_memory_stats()
    
    # Force CUDA context initialization
    _ = torch.zeros(1, device="cuda")
    del _
    torch.cuda.synchronize()
    
    after = get_memory_stats()
    
    context_overhead = after["total"] - after["free"] - before.get("allocated", 0)
    
    result = {
        "cuda_context_gb": round(context_overhead, 3),
        "total_gpu_gb": after["total"],
        "free_after_context_gb": after["free"],
    }
    
    print(f"  CUDA context overhead: {result['cuda_context_gb']:.3f} GB")
    print(f"  Total GPU memory: {result['total_gpu_gb']:.3f} GB")
    print(f"  Free after context: {result['free_after_context_gb']:.3f} GB")
    
    return result


def profile_dit(checkpoint_dir: str, config_path: str = "acestep-v15-turbo") -> Dict[str, Any]:
    """Profile DiT model memory consumption."""
    print("\n" + "=" * 60)
    print(f"Profiling DiT model: {config_path}")
    print("=" * 60)
    
    from transformers import AutoModel
    
    model_path = os.path.join(checkpoint_dir, config_path)
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return {}
    
    reset_memory()
    before = get_memory_stats()
    
    # Load model weights
    print("  Loading DiT model weights...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    torch.cuda.synchronize()
    
    after_load = get_memory_stats()
    weights_gb = after_load["allocated"] - before["allocated"]
    
    print(f"  DiT model weights: {weights_gb:.3f} GB")
    
    # Load silence latent
    silence_path = os.path.join(model_path, "silence_latent.pt")
    silence_latent = None
    if os.path.exists(silence_path):
        silence_latent = torch.load(silence_path, weights_only=True).transpose(1, 2)
        silence_latent = silence_latent.to("cuda").to(torch.bfloat16)
    
    # Determine if model has CFG (base vs turbo)
    has_cfg = "turbo" not in config_path.lower()
    
    # Profile inference at different batch sizes and durations
    inference_results = []
    
    # Duration -> latent_length mapping: 48000 Hz audio, 5 Hz latent = 9600 audio samples per latent frame
    # Actually: latent_length = ceil(duration * 5) for 5Hz models
    durations = [60, 120, 240]
    batch_sizes = [1, 2, 4]
    
    for duration in durations:
        for batch_size in batch_sizes:
            reset_memory()
            torch.cuda.reset_peak_memory_stats()
            
            # Reload model to GPU if needed
            model = model.to("cuda")
            torch.cuda.synchronize()
            
            mem_before_inference = get_memory_stats()
            
            latent_length = int(duration * 5)  # 5 Hz
            latent_dim = 64  # Standard latent dim
            
            try:
                with torch.inference_mode():
                    # Simulate DiT inference inputs
                    # Create dummy latent noise
                    noise = torch.randn(batch_size, latent_length, latent_dim, device="cuda", dtype=torch.bfloat16)
                    
                    # Simulate text encoder output
                    text_hidden = torch.randn(batch_size, 512, 768, device="cuda", dtype=torch.bfloat16)
                    text_mask = torch.ones(batch_size, 512, device="cuda", dtype=torch.long)
                    
                    # If has CFG, double the batch for classifier-free guidance
                    if has_cfg:
                        noise_cfg = torch.cat([noise, noise], dim=0)
                        text_hidden_cfg = torch.cat([text_hidden, text_hidden], dim=0)
                        text_mask_cfg = torch.cat([text_mask, text_mask], dim=0)
                        del noise_cfg, text_hidden_cfg, text_mask_cfg
                    
                    del noise, text_hidden, text_mask
                    torch.cuda.synchronize()
                    
                mem_after_inference = get_memory_stats()
                peak_gb = mem_after_inference["max_allocated"] - mem_before_inference["allocated"]
                
                result_entry = {
                    "duration_s": duration,
                    "batch_size": batch_size,
                    "has_cfg": has_cfg,
                    "peak_inference_gb": round(peak_gb, 3),
                    "latent_length": latent_length,
                }
                inference_results.append(result_entry)
                
                print(f"  batch={batch_size}, dur={duration}s: peak={peak_gb:.3f} GB (cfg={has_cfg})")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  batch={batch_size}, dur={duration}s: OOM")
                    inference_results.append({
                        "duration_s": duration,
                        "batch_size": batch_size,
                        "has_cfg": has_cfg,
                        "peak_inference_gb": -1,
                        "error": "OOM",
                    })
                    torch.cuda.empty_cache()
                else:
                    raise
    
    # Cleanup
    del model
    if silence_latent is not None:
        del silence_latent
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "config_path": config_path,
        "weights_gb": round(weights_gb, 3),
        "has_cfg": has_cfg,
        "inference_results": inference_results,
    }


def profile_vae(checkpoint_dir: str) -> Dict[str, Any]:
    """Profile VAE model memory consumption."""
    print("\n" + "=" * 60)
    print("Profiling VAE model")
    print("=" * 60)
    
    from diffusers.models import AutoencoderOobleck
    
    vae_path = os.path.join(checkpoint_dir, "vae")
    if not os.path.exists(vae_path):
        print(f"  VAE not found: {vae_path}")
        return {}
    
    reset_memory()
    before = get_memory_stats()
    
    # Load VAE
    print("  Loading VAE model weights...")
    vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae = vae.to("cuda").to(torch.float16)
    vae.eval()
    torch.cuda.synchronize()
    
    after_load = get_memory_stats()
    weights_gb = after_load["allocated"] - before["allocated"]
    
    print(f"  VAE model weights: {weights_gb:.3f} GB")
    
    # Profile decode at different chunk sizes
    decode_results = []
    chunk_sizes = [256, 512, 1024]
    
    for chunk_size in chunk_sizes:
        reset_memory()
        torch.cuda.reset_peak_memory_stats()
        
        vae = vae.to("cuda")
        torch.cuda.synchronize()
        
        mem_before = get_memory_stats()
        
        try:
            with torch.inference_mode():
                # Simulate latent input: [batch=1, channels=64, length=chunk_size]
                latent = torch.randn(1, 64, chunk_size, device="cuda", dtype=torch.float16)
                decoder_output = vae.decode(latent)
                audio = decoder_output.sample
                del decoder_output, audio, latent
                torch.cuda.synchronize()
            
            mem_after = get_memory_stats()
            peak_gb = mem_after["max_allocated"] - mem_before["allocated"]
            
            decode_results.append({
                "chunk_size": chunk_size,
                "peak_decode_gb": round(peak_gb, 3),
            })
            print(f"  chunk_size={chunk_size}: peak={peak_gb:.3f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  chunk_size={chunk_size}: OOM")
                decode_results.append({
                    "chunk_size": chunk_size,
                    "peak_decode_gb": -1,
                    "error": "OOM",
                })
                torch.cuda.empty_cache()
            else:
                raise
    
    # Cleanup
    del vae
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "weights_gb": round(weights_gb, 3),
        "decode_results": decode_results,
    }


def profile_text_encoder(checkpoint_dir: str) -> Dict[str, Any]:
    """Profile text encoder memory consumption."""
    print("\n" + "=" * 60)
    print("Profiling Text Encoder")
    print("=" * 60)
    
    from transformers import AutoModel, AutoTokenizer
    
    encoder_path = os.path.join(checkpoint_dir, "text_encoder")
    if not os.path.exists(encoder_path):
        print(f"  Text encoder not found: {encoder_path}")
        return {}
    
    reset_memory()
    before = get_memory_stats()
    
    # Load text encoder
    print("  Loading text encoder weights...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path)
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    torch.cuda.synchronize()
    
    after_load = get_memory_stats()
    weights_gb = after_load["allocated"] - before["allocated"]
    
    print(f"  Text encoder weights: {weights_gb:.3f} GB")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "weights_gb": round(weights_gb, 3),
    }


def profile_lm(checkpoint_dir: str, lm_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Profile LM model memory consumption."""
    print("\n" + "=" * 60)
    print("Profiling 5Hz LM models")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if lm_models is None:
        # Auto-detect available LM models
        lm_models = []
        for name in os.listdir(checkpoint_dir):
            if "5Hz-lm" in name and os.path.isdir(os.path.join(checkpoint_dir, name)):
                lm_models.append(name)
    
    if not lm_models:
        print("  No LM models found")
        return {}
    
    lm_models.sort()
    results = {}
    
    for lm_name in lm_models:
        lm_path = os.path.join(checkpoint_dir, lm_name)
        if not os.path.exists(lm_path):
            print(f"  LM model not found: {lm_path}")
            continue
        
        print(f"\n  Profiling LM: {lm_name}")
        
        reset_memory()
        before = get_memory_stats()
        
        # Load model weights
        print(f"    Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            lm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to("cuda")
        model.eval()
        torch.cuda.synchronize()
        
        after_load = get_memory_stats()
        weights_gb = after_load["allocated"] - before["allocated"]
        
        print(f"    Model weights: {weights_gb:.3f} GB")
        
        # Estimate KV cache memory for different max_model_len values
        # KV cache formula: 2 * num_layers * max_tokens * num_kv_heads * head_dim * dtype_size
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dtype_size = 2  # bfloat16 = 2 bytes
        
        kv_cache_estimates = {}
        for max_len in [2048, 4096]:
            # Per-token KV cache size
            per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_size
            total_bytes = per_token_bytes * max_len
            total_gb = total_bytes / (1024**3)
            kv_cache_estimates[str(max_len)] = round(total_gb, 3)
            print(f"    KV cache ({max_len} tokens): {total_gb:.3f} GB")
        
        results[lm_name] = {
            "weights_gb": round(weights_gb, 3),
            "kv_cache_estimates": kv_cache_estimates,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VRAM Profiling for ACE-Step 1.5")
    parser.add_argument("--component", type=str, default="all",
                       choices=["all", "cuda_context", "dit", "vae", "text_encoder", "lm"],
                       help="Component to profile (default: all)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Checkpoint directory (default: auto-detect)")
    parser.add_argument("--dit-config", type=str, default="acestep-v15-turbo",
                       help="DiT model config name (default: acestep-v15-turbo)")
    parser.add_argument("--lm-models", type=str, nargs="*", default=None,
                       help="LM models to profile (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a CUDA GPU.")
        sys.exit(1)
    
    # Auto-detect checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print("=" * 60)
    print("ACE-Step 1.5 VRAM Profiler")
    print("=" * 60)
    print(f"  GPU: {device_name}")
    print(f"  Total VRAM: {total_mem:.2f} GB")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Component: {args.component}")
    
    results = {
        "gpu_name": device_name,
        "total_vram_gb": round(total_mem, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    components = [args.component] if args.component != "all" else [
        "cuda_context", "dit", "vae", "text_encoder", "lm"
    ]
    
    for component in components:
        if component == "cuda_context":
            results["cuda_context"] = measure_cuda_context()
        elif component == "dit":
            results["dit"] = profile_dit(args.checkpoint_dir, args.dit_config)
        elif component == "vae":
            results["vae"] = profile_vae(args.checkpoint_dir)
        elif component == "text_encoder":
            results["text_encoder"] = profile_text_encoder(args.checkpoint_dir)
        elif component == "lm":
            results["lm"] = profile_lm(args.checkpoint_dir, args.lm_models)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if "cuda_context" in results:
        print(f"  CUDA context: {results['cuda_context'].get('cuda_context_gb', 'N/A')} GB")
    if "dit" in results and results["dit"]:
        print(f"  DiT weights ({results['dit'].get('config_path', '')}): {results['dit'].get('weights_gb', 'N/A')} GB")
    if "vae" in results and results["vae"]:
        print(f"  VAE weights: {results['vae'].get('weights_gb', 'N/A')} GB")
    if "text_encoder" in results and results["text_encoder"]:
        print(f"  Text encoder weights: {results['text_encoder'].get('weights_gb', 'N/A')} GB")
    if "lm" in results and results["lm"]:
        for lm_name, lm_data in results["lm"].items():
            print(f"  LM {lm_name} weights: {lm_data.get('weights_gb', 'N/A')} GB")
    
    # Calculate total base VRAM (all models loaded simultaneously)
    base_total = 0
    if "cuda_context" in results:
        base_total += results["cuda_context"].get("cuda_context_gb", 0)
    if "dit" in results and results["dit"]:
        base_total += results["dit"].get("weights_gb", 0)
    if "vae" in results and results["vae"]:
        base_total += results["vae"].get("weights_gb", 0)
    if "text_encoder" in results and results["text_encoder"]:
        base_total += results["text_encoder"].get("weights_gb", 0)
    
    print(f"\n  Base VRAM (DiT+VAE+TextEnc+CUDA): {base_total:.3f} GB")
    print(f"  Remaining for LM + inference: {total_mem - base_total:.3f} GB")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(PROJECT_ROOT, "scripts", "vram_profile_results.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
