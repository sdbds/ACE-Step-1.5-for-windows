"""
Wizard flow builders for each subcommand (train, preprocess, estimate).

Each function collects user input via prompt helpers and returns a populated
``argparse.Namespace`` identical to what the CLI would produce.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    IS_WINDOWS,
    DEFAULT_NUM_WORKERS,
    menu,
    ask,
    ask_path,
    ask_bool,
    section,
)


def wizard_train(mode: str) -> argparse.Namespace:
    """Interactive wizard for training (fixed or vanilla)."""

    config_mode = menu(
        "How much do you want to configure?",
        [
            ("basic", "Basic (recommended defaults, fewer questions)"),
            ("advanced", "Advanced (all settings exposed)"),
        ],
        default=1,
    )

    section("Required Settings")
    checkpoint_dir = ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])
    dataset_dir = ask_path("Dataset directory (preprocessed .pt files)", must_exist=True)
    output_dir = ask("Output directory for LoRA weights", required=True)

    section("LoRA Settings (press Enter for defaults)")
    rank = ask("Rank", default=64, type_fn=int)
    alpha = ask("Alpha", default=128, type_fn=int)
    dropout = ask("Dropout", default=0.1, type_fn=float)

    attention_type = menu(
        "Which attention layers to target?",
        [
            ("both", "Both self-attention and cross-attention"),
            ("self", "Self-attention only (audio patterns)"),
            ("cross", "Cross-attention only (text conditioning)"),
        ],
        default=1,
    )

    target_modules_str = ask("Target projections", default="q_proj k_proj v_proj o_proj")
    target_modules = target_modules_str.split()

    section("Training Settings (press Enter for defaults)")
    learning_rate = ask("Learning rate", default=1e-4, type_fn=float)
    batch_size = ask("Batch size", default=1, type_fn=int)
    gradient_accumulation = ask("Gradient accumulation", default=4, type_fn=int)
    epochs = ask("Max epochs", default=100, type_fn=int)
    warmup_steps = ask("Warmup steps", default=100, type_fn=int)
    seed = ask("Seed", default=42, type_fn=int)

    cfg_ratio = 0.15
    if mode == "fixed":
        section("Corrected Training Settings")
        cfg_ratio = ask("CFG dropout ratio", default=0.15, type_fn=float)

    section("Logging & Checkpoints (press Enter for defaults)")
    save_every = ask("Save checkpoint every N epochs", default=10, type_fn=int)
    log_every = ask("Log metrics every N steps", default=10, type_fn=int)
    resume_from = ask("Resume from checkpoint path (leave empty to skip)", default=None)
    if resume_from == "None" or resume_from == "":
        resume_from = None

    # ---- Advanced mode settings (defaults used in basic mode) ----
    device = "auto"
    precision = "auto"
    weight_decay = 0.01
    max_grad_norm = 1.0
    bias = "none"
    num_workers = DEFAULT_NUM_WORKERS
    pin_memory = True
    prefetch_factor = 2 if num_workers > 0 else 0
    persistent_workers = num_workers > 0
    log_dir = None
    log_heavy_every = 50
    sample_every_n_epochs = 0
    optimizer_type = "adamw"
    scheduler_type = "cosine"
    gradient_checkpointing = False
    offload_encoder = False

    if config_mode == "advanced":
        section("Device & Precision (Advanced)")
        device = ask("Device", default="auto", choices=["auto", "cuda", "cuda:0", "cuda:1", "mps", "xpu", "cpu"])
        precision = ask("Precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

        section("Optimizer & Scheduler")
        optimizer_type = menu(
            "Which optimizer to use?",
            [
                ("adamw", "AdamW (default, reliable)"),
                ("adamw8bit", "AdamW 8-bit (saves ~30% optimizer VRAM, needs bitsandbytes)"),
                ("adafactor", "Adafactor (minimal state memory)"),
                ("prodigy", "Prodigy (auto-tunes LR -- set LR to 1.0, needs prodigyopt)"),
            ],
            default=1,
        )
        if optimizer_type == "prodigy":
            learning_rate = ask("Learning rate (Prodigy: use 1.0)", default=1.0, type_fn=float)

        scheduler_type = menu(
            "LR scheduler?",
            [
                ("cosine", "Cosine Annealing (gradual decay, most popular)"),
                ("linear", "Linear (steady decay to near-zero)"),
                ("constant", "Constant (flat LR after warmup)"),
                ("constant_with_warmup", "Constant with Warmup (explicit warmup then flat)"),
            ],
            default=1,
        )

        section("VRAM Savings (Advanced)")
        gradient_checkpointing = ask_bool(
            "Enable gradient checkpointing? (saves ~40-60% activation VRAM, ~30% slower)",
            default=False,
        )
        offload_encoder = ask_bool(
            "Offload encoder/VAE to CPU? (saves ~2-4GB VRAM after setup)",
            default=False,
        )

        section("Advanced Training Settings")
        weight_decay = ask("Weight decay (L2 regularization)", default=0.01, type_fn=float)
        max_grad_norm = ask("Max gradient norm (clipping)", default=1.0, type_fn=float)
        bias = ask("Bias training mode", default="none", choices=["none", "all", "lora_only"])

        section("Data Loading (Advanced)")
        num_workers = ask("DataLoader workers", default=DEFAULT_NUM_WORKERS, type_fn=int)
        if IS_WINDOWS and num_workers > 0:
            if is_rich_active() and console is not None:
                console.print("  [yellow]Warning: Windows detected -- forcing num_workers=0 (spawn incompatible)[/]")
            else:
                print("  Warning: Windows detected -- forcing num_workers=0 (spawn incompatible)")
            num_workers = 0
        pin_memory = ask_bool("Pin memory for GPU transfer?", default=True)
        prefetch_factor = ask("Prefetch factor", default=2 if num_workers > 0 else 0, type_fn=int)
        persistent_workers = ask_bool("Keep workers alive between epochs?", default=num_workers > 0)

        section("Advanced Logging")
        log_dir = ask("TensorBoard log directory (leave empty for default)", default=None)
        if log_dir == "None" or log_dir == "":
            log_dir = None
        log_heavy_every = ask("Log gradient norms every N steps", default=50, type_fn=int)
        sample_every_n_epochs = ask("Generate sample every N epochs (0=disabled)", default=0, type_fn=int)

    return argparse.Namespace(
        subcommand=mode,
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device=device,
        precision=precision,
        dataset_dir=dataset_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        seed=seed,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        attention_type=attention_type,
        bias=bias,
        output_dir=output_dir,
        save_every=save_every,
        resume_from=resume_from,
        log_dir=log_dir,
        log_every=log_every,
        log_heavy_every=log_heavy_every,
        sample_every_n_epochs=sample_every_n_epochs,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        gradient_checkpointing=gradient_checkpointing,
        offload_encoder=offload_encoder,
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=240.0,
        cfg_ratio=cfg_ratio,
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )


def wizard_preprocess() -> argparse.Namespace:
    """Interactive wizard for preprocessing."""
    section("Preprocessing Settings")
    checkpoint_dir = ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])

    # Ask about the metadata JSON first so the user understands what
    # information will (or won't) be available during preprocessing.
    if is_rich_active() and console is not None:
        console.print(
            "\n  [dim]A dataset JSON provides lyrics, genre, BPM, key, and "
            "other metadata for each audio file, plus audio file paths.\n"
            "  Without it, you will be asked for an audio directory and all "
            "tracks will default to [Instrumental] with no genre/BPM info.[/]"
        )
    else:
        print(
            "\n  A dataset JSON provides lyrics, genre, BPM, key, and "
            "other metadata for each audio file, plus audio file paths.\n"
            "  Without it, you will be asked for an audio directory and all "
            "tracks will default to [Instrumental] with no genre/BPM info."
        )
    dataset_json = ask("Dataset JSON file (leave empty to skip)", default=None)
    if dataset_json == "None" or dataset_json == "":
        dataset_json = None
    elif dataset_json:
        # Resolve the path (handles missing leading ./ or /)
        json_path = Path(dataset_json).resolve()
        if not json_path.is_file():
            if is_rich_active() and console is not None:
                console.print(
                    f"  [yellow]Not found: {dataset_json}[/]\n"
                    f"  [dim]Falling back to audio directory scan.[/]"
                )
            else:
                print(f"  Not found: {dataset_json}")
                print("  Falling back to audio directory scan.")
            dataset_json = None
        else:
            dataset_json = str(json_path)

    # Only ask for audio directory when no (valid) JSON was provided.
    # With a JSON, audio paths come from the file itself.
    if not dataset_json:
        if is_rich_active() and console is not None:
            console.print(
                "  [dim]Subdirectories will be scanned recursively.[/]"
            )
        else:
            print("  Subdirectories will be scanned recursively.")
        audio_dir = ask_path("Audio directory (source audio files)", must_exist=True)
    else:
        audio_dir = None

    tensor_output = ask("Output directory for .pt tensor files", required=True)
    max_duration = ask("Max audio duration in seconds", default=240.0, type_fn=float)

    return argparse.Namespace(
        subcommand="fixed",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device="auto",
        precision="auto",
        dataset_dir=tensor_output,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if DEFAULT_NUM_WORKERS > 0 else 0,
        persistent_workers=DEFAULT_NUM_WORKERS > 0,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        epochs=100,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        rank=64,
        alpha=128,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        attention_type="both",
        bias="none",
        output_dir="./lora_output",
        save_every=10,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        sample_every_n_epochs=0,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=False,
        offload_encoder=False,
        preprocess=True,
        audio_dir=audio_dir,
        dataset_json=dataset_json,
        tensor_output=tensor_output,
        max_duration=max_duration,
        cfg_ratio=0.15,
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )


def wizard_estimate() -> argparse.Namespace:
    """Interactive wizard for gradient sensitivity estimation."""
    section("Gradient Sensitivity Estimation")

    if is_rich_active() and console is not None:
        console.print(
            "  [dim]Estimates which LoRA layers learn fastest for your dataset.\n"
            "  Results are saved as JSON and can be used to guide rank selection.[/]\n"
        )

    checkpoint_dir = ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])
    dataset_dir = ask_path("Dataset directory (preprocessed .pt files)", must_exist=True)

    section("Estimation Parameters")
    estimate_batches = ask("Number of estimation batches", default=5, type_fn=int)
    top_k = ask("Top-K layers to highlight", default=16, type_fn=int)
    granularity = ask("Granularity", default="module", choices=["module", "layer"])

    section("LoRA Settings (for estimation)")
    rank = ask("Rank", default=64, type_fn=int)
    alpha = ask("Alpha", default=128, type_fn=int)
    dropout = ask("Dropout", default=0.1, type_fn=float)

    estimate_output = ask("Output JSON path", default="./estimate_results.json")

    return argparse.Namespace(
        subcommand="estimate",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device="auto",
        precision="auto",
        dataset_dir=dataset_dir,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if DEFAULT_NUM_WORKERS > 0 else 0,
        persistent_workers=DEFAULT_NUM_WORKERS > 0,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        epochs=1,
        warmup_steps=0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        attention_type="both",
        bias="none",
        output_dir="./estimate_output",
        save_every=999,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        sample_every_n_epochs=0,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=False,
        offload_encoder=False,
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=240.0,
        cfg_ratio=0.15,
        estimate_batches=estimate_batches,
        top_k=top_k,
        granularity=granularity,
        module_config=None,
        auto_estimate=False,
        estimate_output=estimate_output,
    )
