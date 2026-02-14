"""
Config-object construction for ACE-Step Training V2 CLI.

Reads model ``config.json`` for timestep parameters, auto-detects GPU,
and builds ``LoRAConfigV2`` / ``TrainingConfigV2`` from parsed CLI args.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
from acestep.training_v2.gpu_utils import detect_gpu
from acestep.training_v2.cli.args import VARIANT_DIR_MAP
from acestep.training_v2.cli.validation import resolve_target_modules

logger = logging.getLogger(__name__)


def build_configs(args: argparse.Namespace) -> Tuple[LoRAConfigV2, TrainingConfigV2]:
    """Construct LoRAConfigV2 and TrainingConfigV2 from parsed CLI args.

    Also patches in ``timestep_mu``, ``timestep_sigma``, and
    ``data_proportion`` from the model's ``config.json`` so the user
    does not need to pass them manually.
    """
    import json as _json

    # -- Resolve model config path ------------------------------------------
    ckpt_root = Path(args.checkpoint_dir)
    variant_dir = VARIANT_DIR_MAP[args.model_variant]
    model_config_path = ckpt_root / variant_dir / "config.json"

    timestep_mu = -0.4
    timestep_sigma = 1.0
    data_proportion = 0.5

    if model_config_path.is_file():
        try:
            mcfg = _json.loads(model_config_path.read_text())
            timestep_mu = mcfg.get("timestep_mu", timestep_mu)
            timestep_sigma = mcfg.get("timestep_sigma", timestep_sigma)
            data_proportion = mcfg.get("data_proportion", data_proportion)
        except (_json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[Side-Step] Failed to parse %s: %s -- using default timestep parameters",
                model_config_path, exc,
            )

    # -- GPU info -----------------------------------------------------------
    gpu_info = detect_gpu(
        requested_device=args.device,
        requested_precision=args.precision,
    )

    # -- LoRA config --------------------------------------------------------
    attention_type = getattr(args, "attention_type", "both")
    resolved_modules = resolve_target_modules(args.target_modules, attention_type)

    lora_cfg = LoRAConfigV2(
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=resolved_modules,
        bias=args.bias,
        attention_type=attention_type,
    )

    # -- Clamp DataLoader flags when num_workers <= 0 -----------------------
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor
    persistent_workers = args.persistent_workers

    if num_workers <= 0:
        if persistent_workers:
            logger.info("[Side-Step] num_workers=0 -- forcing persistent_workers=False")
            persistent_workers = False
        if prefetch_factor and prefetch_factor > 0:
            logger.info("[Side-Step] num_workers=0 -- forcing prefetch_factor=0")
            prefetch_factor = 0

    # -- Training config ----------------------------------------------------
    train_cfg = TrainingConfigV2(
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
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        # V2 extensions
        optimizer_type=getattr(args, "optimizer_type", "adamw"),
        scheduler_type=getattr(args, "scheduler_type", "cosine"),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        offload_encoder=getattr(args, "offload_encoder", False),
        cfg_ratio=getattr(args, "cfg_ratio", 0.15),
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        model_variant=args.model_variant,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=args.resume_from,
        log_dir=args.log_dir,
        log_every=args.log_every,
        log_heavy_every=args.log_heavy_every,
        sample_every_n_epochs=args.sample_every_n_epochs,
        # Estimation / selective (may not exist on all subcommands)
        estimate_batches=getattr(args, "estimate_batches", None),
        top_k=getattr(args, "top_k", 16),
        granularity=getattr(args, "granularity", "module"),
        module_config=getattr(args, "module_config", None),
        auto_estimate=getattr(args, "auto_estimate", False),
        estimate_output=getattr(args, "estimate_output", None),
        # Preprocessing
        preprocess=args.preprocess,
        audio_dir=args.audio_dir,
        dataset_json=args.dataset_json,
        tensor_output=args.tensor_output,
        max_duration=args.max_duration,
    )

    return lora_cfg, train_cfg
