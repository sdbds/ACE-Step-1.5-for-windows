"""
vanilla subcommand -- Reproduce existing (bugged) training for backward compatibility.

Imports the original ``PreprocessedLoRAModule`` and ``LoRATrainer`` from
``acestep/training/``.  Always prints a warning that behaviour differs from
the model's own training procedure.

This module is never modified and is intentionally thin: it just wires CLI
args to the original trainer API.
"""

from __future__ import annotations

import argparse

import torch

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.trainer import LoRATrainer
from acestep.training_v2.gpu_utils import detect_gpu
from acestep.training_v2.model_loader import load_decoder_for_training


# ---------------------------------------------------------------------------
# Handler shim
# ---------------------------------------------------------------------------

class _HandlerShim:
    """Minimal shim satisfying the ``LoRATrainer`` constructor.

    The original trainer expects a ``dit_handler`` with ``.model``,
    ``.device``, ``.dtype``, and ``.quantization``.
    """

    def __init__(self, model: torch.nn.Module, device: str, dtype: torch.dtype) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.quantization = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_vanilla(args: argparse.Namespace) -> int:
    """Execute the vanilla training subcommand.

    Returns 0 on success, non-zero on failure.
    """
    # -- UI setup -------------------------------------------------------------
    from acestep.training_v2.ui import set_plain_mode
    from acestep.training_v2.ui.banner import show_banner
    from acestep.training_v2.ui.errors import handle_error, show_info, show_warning
    from acestep.training_v2.ui.progress import track_training
    from acestep.training_v2.ui.summary import show_summary

    if getattr(args, "plain", False):
        set_plain_mode(True)

    # -- Matmul precision (matches handler.initialize_service behaviour) ------
    torch.set_float32_matmul_precision("medium")

    # -- GPU detection -------------------------------------------------------
    gpu = detect_gpu(args.device, args.precision)

    # -- Extra-strong warning for base/sft + vanilla (especially problematic) --
    if args.model_variant in ("base", "sft"):
        from acestep.training_v2.ui.config_panel import confirm_start
        from acestep.training_v2.ui.errors import show_error

        show_error(
            title="Vanilla + Non-Turbo Model Warning",
            message=(
                f"You're using vanilla mode with --model-variant {args.model_variant}.\n\n"
                "The vanilla trainer uses turbo's 8-step discrete timesteps, which is:\n"
                "  1. Wrong for training (should use continuous logit-normal sampling)\n"
                "  2. Extra wrong for base/sft (they use different inference schedules)\n\n"
                "This combination will likely produce poor results.\n"
                "Use 'fixed' mode instead for proper training behavior."
            ),
            suggestion=f"python train.py fixed --model-variant {args.model_variant} ...",
        )
        if not getattr(args, "yes", False):
            if not confirm_start(skip=False):
                show_info("Aborted. Use 'fixed' mode for correct training.")
                return 0

    # -- Banner & warning (skip if wizard already showed them) ----------------
    if not getattr(args, "_from_wizard", False):
        show_warning(
            "vanilla mode reproduces the EXISTING training behaviour which\n"
            "         differs from the model's own training procedure:\n"
            "           - Discrete 8-step turbo timesteps (should be continuous logit-normal)\n"
            "           - No CFG dropout (should be cfg_ratio=0.15)\n"
            "         These bugs affect ALL model variants (turbo, base, sft).\n"
            "         Use 'fixed' or 'selective' for corrected training."
        )
        show_banner(
            subcommand="vanilla",
            device=gpu.device,
            precision=gpu.precision,
        )

    # -- Build original config objects from CLI args -------------------------
    lora_cfg = LoRAConfig(
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
        bias=args.bias,
    )
    train_cfg = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every_n_epochs=args.save_every,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        log_every_n_steps=args.log_every,
    )

    # -- Load model ----------------------------------------------------------
    try:
        show_info(f"Loading model (variant={args.model_variant}, device={gpu.device})")
        model = load_decoder_for_training(
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            device=gpu.device,
            precision=gpu.precision,
        )
    except Exception as exc:
        handle_error(exc, context="Model loading", show_traceback=True)
        return 1

    # -- Build handler shim --------------------------------------------------
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    handler = _HandlerShim(
        model=model,
        device=gpu.device,
        dtype=dtype_map.get(gpu.precision, torch.bfloat16),
    )

    # -- Run original trainer ------------------------------------------------
    try:
        trainer = LoRATrainer(handler, lora_cfg, train_cfg)

        stats = track_training(
            training_iter=trainer.train_from_preprocessed(
                tensor_dir=args.dataset_dir,
                resume_from=args.resume_from,
            ),
            max_epochs=args.epochs,
            device=gpu.device,
        )

        # -- Summary ----------------------------------------------------------
        show_summary(
            stats=stats,
            output_dir=args.output_dir,
            log_dir=None,  # vanilla doesn't have configurable log_dir
        )
    except KeyboardInterrupt:
        show_info("Training interrupted by user (Ctrl+C)")
        return 130
    except Exception as exc:
        handle_error(exc, context="Training", show_traceback=True)
        return 1

    return 0
