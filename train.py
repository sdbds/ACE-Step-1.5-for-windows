#!/usr/bin/env python3
"""
ACE-Step Training V2 -- CLI Entry Point

Usage:
    python train.py <subcommand> [args]

Subcommands:
    vanilla          Reproduce existing (bugged) training for backward compatibility
    fixed            Corrected training: continuous timesteps + CFG dropout
    selective        Corrected training with dataset-specific module selection
    estimate         Gradient sensitivity analysis (no training)
    compare-configs  Compare module config JSON files

Examples:
    python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo \\
        --dataset-dir ./preprocessed_tensors/jazz --output-dir ./lora_output/jazz

    python train.py --help
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console (same as before)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

# File (captures DEBUG+ including tracebacks)
_file_handler = logging.FileHandler("sidestep.log", mode="a", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
logger = logging.getLogger("train")


def _has_subcommand() -> bool:
    """Check if sys.argv contains a recognized subcommand or --help."""
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True  # let argparse handle help
    known = {"vanilla", "fixed", "selective", "estimate", "compare-configs"}
    return bool(known & set(args))


def main() -> int:
    # -- Compatibility check (non-fatal) ------------------------------------
    try:
        from acestep.training_v2._compat import check_compatibility
        check_compatibility()
    except Exception:
        pass  # never let the compat check itself crash the CLI

    # -- Interactive wizard when no subcommand is given -----------------------
    if not _has_subcommand():
        from acestep.training_v2.ui.wizard import run_wizard

        args = run_wizard()
        if args is None:
            return 0
    else:
        from acestep.training_v2.cli.common import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args()

    from acestep.training_v2.cli.common import validate_paths

    # -- Preprocessing (wizard flow) ------------------------------------------
    if getattr(args, "preprocess", False):
        return _run_preprocess(args)

    # -- Dispatch -----------------------------------------------------------
    sub = args.subcommand

    # compare-configs has its own validation
    if sub == "compare-configs":
        return _run_compare_configs(args)

    # All other subcommands need path validation
    if not validate_paths(args):
        return 1

    if sub == "vanilla":
        from acestep.training_v2.cli.train_vanilla import run_vanilla
        return run_vanilla(args)

    elif sub == "fixed":
        from acestep.training_v2.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "selective":
        return _run_selective(args)

    elif sub == "estimate":
        return _run_estimate(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


# ===========================================================================
# Subcommand implementations
# ===========================================================================

def _run_preprocess(args) -> int:
    """Run the two-pass preprocessing pipeline."""
    from acestep.training_v2.preprocess import preprocess_audio_files

    audio_dir = getattr(args, "audio_dir", None)
    dataset_json = getattr(args, "dataset_json", None)
    tensor_output = getattr(args, "tensor_output", None)

    if not audio_dir and not dataset_json:
        print("[FAIL] --audio-dir or --dataset-json is required for preprocessing.", file=sys.stderr)
        return 1
    if not tensor_output:
        print("[FAIL] --tensor-output is required for preprocessing.", file=sys.stderr)
        return 1

    source_label = dataset_json if dataset_json else audio_dir
    print(f"[INFO] Preprocessing: {source_label} -> {tensor_output}")
    print(f"[INFO] Two-pass pipeline (sequential model loading for low VRAM)")

    try:
        result = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=tensor_output,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=getattr(args, "max_duration", 240.0),
            dataset_json=dataset_json,
            device=getattr(args, "device", "auto"),
            precision=getattr(args, "precision", "auto"),
        )
    except Exception as exc:
        print(f"[FAIL] Preprocessing failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing error")
        return 1

    print(f"\n[OK] Preprocessing complete:")
    print(f"     Processed: {result['processed']}/{result['total']}")
    if result["failed"]:
        print(f"     Failed:    {result['failed']}")
    print(f"     Output:    {result['output_dir']}")
    print(f"\n[INFO] You can now train with:")
    print(f"       python train.py fixed --dataset-dir {result['output_dir']} ...")
    return 0


def _run_selective(args) -> int:
    """Run selective training (placeholder -- full implementation in Conversation C)."""
    print("[INFO] Selective training is not yet implemented.")
    print("[INFO] Use 'fixed' for corrected training, or 'estimate' for module analysis.")
    return 0


def _run_estimate(args) -> int:
    """Run gradient sensitivity estimation."""
    import json as _json
    from acestep.training_v2.estimate import run_estimation

    num_batches = getattr(args, "estimate_batches", 5) or 5

    print(f"[INFO] Running gradient estimation ({num_batches} batches) ...")
    try:
        results = run_estimation(
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            dataset_dir=args.dataset_dir,
            num_batches=num_batches,
            batch_size=args.batch_size,
            top_k=getattr(args, "top_k", 16) or 16,
            granularity=getattr(args, "granularity", "module") or "module",
        )
    except Exception as exc:
        print(f"[FAIL] Estimation failed: {exc}", file=sys.stderr)
        logger.exception("Estimation error")
        return 1

    if not results:
        print("[WARN] No results -- dataset may be empty or model incompatible.")
        return 1

    # Display results
    print(f"\n{'='*60}")
    print(f"  Top-{len(results)} Modules by Gradient Sensitivity")
    print(f"{'='*60}")
    for i, entry in enumerate(results, 1):
        print(f"  {i:3d}. {entry['module']:<50s}  {entry['sensitivity']:.6f}")
    print(f"{'='*60}\n")

    # Save to JSON
    output_path = getattr(args, "estimate_output", None) or "./estimate_results.json"
    try:
        with open(output_path, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"[OK] Results saved to {output_path}")
    except OSError as exc:
        print(f"[WARN] Could not save results: {exc}", file=sys.stderr)

    return 0


def _run_compare_configs(args) -> int:
    """Compare module config JSON files (placeholder -- full implementation in Conversation D)."""
    from acestep.training_v2.cli.common import validate_paths
    if not validate_paths(args):
        return 1
    print("[INFO] compare-configs is not yet implemented.")
    return 0


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
