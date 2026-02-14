"""
FixedLoRAModule + FixedLoRATrainer -- Corrected Training for ACE-Step V2

Differences from the original trainer (``acestep/training/trainer.py``):
    1. **Continuous logit-normal timestep sampling** via ``sample_timesteps()``
       instead of discrete 8-step turbo schedule.
    2. **CFG dropout** (``cfg_ratio=0.15``) replacing conditions with
       ``model.null_condition_emb`` -- the original trainer had none.

Reuses unchanged utilities from ``acestep/training/``:
    - ``inject_lora_into_dit``, ``save_lora_weights``,
      ``save_training_checkpoint``, ``load_training_checkpoint``
    - ``PreprocessedDataModule``
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from acestep.training_v2.optim import build_optimizer, build_scheduler

# Reuse existing utilities (never modified)
from acestep.training.lora_utils import (
    check_peft_available,
    inject_lora_into_dit,
    load_training_checkpoint,
    save_lora_weights,
    save_training_checkpoint,
)
from acestep.training.data_module import PreprocessedDataModule

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.ui import TrainingUpdate

logger = logging.getLogger(__name__)

# Try to import Lightning Fabric
try:
    from lightning.fabric import Fabric

    _FABRIC_AVAILABLE = True
except ImportError:
    _FABRIC_AVAILABLE = False
    logger.warning("[WARN] Lightning Fabric not installed. Training will use basic loop.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================

class FixedLoRAModule(nn.Module):
    """LoRA training module with corrected timestep sampling and CFG dropout.

    Training flow (per step):
        1. Load pre-computed tensors (from ``PreprocessedDataModule``).
        2. Apply **CFG dropout** on ``encoder_hidden_states``.
        3. Sample noise ``x1`` and continuous timestep ``t`` via
           ``sample_timesteps()`` (logit-normal).
        4. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        5. Forward through decoder, compute flow matching loss.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfigV2,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.lora_config = lora_config
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")

        # -- LoRA injection --------------------------------------------------
        if check_peft_available():
            self.model, self.lora_info = inject_lora_into_dit(model, lora_config)
            logger.info(
                "[OK] LoRA injected: %s trainable params",
                f"{self.lora_info['trainable_params']:,}",
            )
        else:
            self.model = model
            self.lora_info: Dict[str, Any] = {}
            logger.warning("[WARN] PEFT not available -- training without LoRA adapters")

        # Model config (for timestep params read at runtime)
        self.config = model.config

        # -- Null condition embedding for CFG dropout ------------------------
        # ``model.null_condition_emb`` is a Parameter on the top-level model
        # (not the decoder).
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            logger.warning(
                "[WARN] model.null_condition_emb not found -- CFG dropout disabled"
            )

        # -- Timestep sampling params ----------------------------------------
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio

        # Book-keeping
        self.training_losses: List[float] = []

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with corrected timestep sampling + CFG dropout.

        Args:
            batch: Dict with keys ``target_latents``, ``attention_mask``,
                ``encoder_hidden_states``, ``encoder_attention_mask``,
                ``context_latents``.

        Returns:
            Scalar loss tensor (``float32`` for stable backward).
        """
        # Mixed-precision context
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking

            target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
            attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

            bsz = target_latents.shape[0]

            # ---- CFG dropout (CORRECTED -- missing in original trainer) ----
            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

            # ---- Flow matching noise ----------------------------------------
            x1 = torch.randn_like(target_latents)  # noise
            x0 = target_latents  # data

            # ---- Continuous timestep sampling (CORRECTED) -------------------
            t, r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,  # r = t for all ACE-Step variants
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # ---- Interpolate x_t -------------------------------------------
            xt = t_ * x1 + (1.0 - t_) * x0

            # ---- Decoder forward -------------------------------------------
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,  # r = t
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # ---- Flow matching loss ----------------------------------------
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        # fp32 for stable backward
        diffusion_loss = diffusion_loss.float()
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss


# ===========================================================================
# FixedLoRATrainer -- orchestration
# ===========================================================================

class FixedLoRATrainer:
    """High-level trainer for corrected ACE-Step LoRA fine-tuning.

    Uses Lightning Fabric for mixed precision and gradient scaling.
    Falls back to a basic PyTorch loop when Fabric is not installed.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfigV2,
        training_config: TrainingConfigV2,
    ) -> None:
        self.model = model
        self.lora_config = lora_config
        self.training_config = training_config

        self.module: Optional[FixedLoRAModule] = None
        self.fabric: Optional[Any] = None
        self.is_training = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        training_state: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Run the full training loop.

        Yields ``(global_step, loss, status_message)`` tuples.
        """
        self.is_training = True
        cfg = self.training_config

        try:
            # -- Validate ---------------------------------------------------
            ds_dir = Path(cfg.dataset_dir)
            if not ds_dir.is_dir():
                yield TrainingUpdate(0, 0.0, f"[FAIL] Dataset directory not found: {ds_dir}", kind="fail")
                return

            # -- Seed -------------------------------------------------------
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

            # -- Build module -----------------------------------------------
            device = torch.device(cfg.device)
            dtype = _select_compute_dtype(_normalize_device_type(device))

            self.module = FixedLoRAModule(
                model=self.model,
                lora_config=self.lora_config,
                training_config=cfg,
                device=device,
                dtype=dtype,
            )

            # -- Data -------------------------------------------------------
            # Windows uses spawn for multiprocessing; default to 0 workers there
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
                num_workers = 0

            data_module = PreprocessedDataModule(
                tensor_dir=cfg.dataset_dir,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
                persistent_workers=cfg.persistent_workers if num_workers > 0 else False,
            )
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                yield TrainingUpdate(0, 0.0, "[FAIL] No valid samples found in dataset directory", kind="fail")
                return

            yield TrainingUpdate(0, 0.0, f"[OK] Loaded {len(data_module.train_dataset)} preprocessed samples", kind="info")

            # -- Dispatch to Fabric or basic loop ---------------------------
            if _FABRIC_AVAILABLE:
                yield from self._train_fabric(data_module, training_state)
            else:
                yield from self._train_basic(data_module, training_state)

        except Exception as exc:
            logger.exception("Training failed")
            yield TrainingUpdate(0, 0.0, f"[FAIL] Training failed: {exc}", kind="fail")
        finally:
            self.is_training = False

    def stop(self) -> None:
        self.is_training = False

    @staticmethod
    def _offload_non_decoder(model: nn.Module) -> int:
        """Move encoder/VAE/non-decoder submodules to CPU. Returns count offloaded."""
        count = 0
        for name in ("music_encoder", "lyric_encoder", "timbre_encoder",
                      "condition_projection", "vae", "text_encoder", "attention_pooler"):
            sub = getattr(model, name, None)
            if sub is not None and isinstance(sub, nn.Module):
                sub.to("cpu")
                count += 1
        return count

    # ------------------------------------------------------------------
    # Fabric training loop
    # ------------------------------------------------------------------

    def _train_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"

        # -- Fabric init ----------------------------------------------------
        self.fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision=precision,
        )
        self.fabric.launch()

        yield TrainingUpdate(0, 0.0, f"[INFO] Starting training (device: {device_type}, precision: {precision})", kind="info")

        # -- TensorBoard logger ---------------------------------------------
        tb = TrainingLogger(cfg.effective_log_dir)

        # -- Dataloader -----------------------------------------------------
        train_loader = data_module.train_dataloader()

        # -- Trainable params / optimizer -----------------------------------
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        if not trainable_params:
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            tb.close()
            return

        yield TrainingUpdate(0, 0.0, f"[INFO] Training {sum(p.numel() for p in trainable_params):,} parameters", kind="info")

        optimizer_type = getattr(cfg, "optimizer_type", "adamw")
        optimizer = build_optimizer(
            trainable_params,
            optimizer_type=optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device_type=self.module.device.type,
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Optimizer: {optimizer_type}", kind="info")

        # -- Scheduler -------------------------------------------------------
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
        total_steps = steps_per_epoch * cfg.max_epochs

        scheduler_type = getattr(cfg, "scheduler_type", "cosine")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            lr=cfg.learning_rate,
            optimizer_type=optimizer_type,
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Scheduler: {scheduler_type}", kind="info")

        # -- Gradient checkpointing -----------------------------------------
        if getattr(cfg, "gradient_checkpointing", False):
            if hasattr(self.module.model, "decoder") and hasattr(self.module.model.decoder, "gradient_checkpointing_enable"):
                self.module.model.decoder.gradient_checkpointing_enable()
                yield TrainingUpdate(0, 0.0, "[INFO] Gradient checkpointing enabled (saves VRAM, slower)", kind="info")
            else:
                yield TrainingUpdate(0, 0.0, "[WARN] Gradient checkpointing not supported by this model", kind="warning")

        # -- Encoder/VAE offloading ------------------------------------------
        if getattr(cfg, "offload_encoder", False):
            offloaded = self._offload_non_decoder(self.module.model)
            if offloaded:
                yield TrainingUpdate(0, 0.0, f"[INFO] Offloaded {offloaded} model components to CPU (saves VRAM)", kind="info")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -- dtype / Fabric setup -------------------------------------------
        self.module.model = self.module.model.to(self.module.dtype)
        self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)

        # -- Resume ---------------------------------------------------------
        start_epoch = 0
        global_step = 0

        if cfg.resume_from and Path(cfg.resume_from).exists():
            try:
                yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
                ckpt_info = load_training_checkpoint(
                    cfg.resume_from,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                )
                if ckpt_info["adapter_path"]:
                    adapter_path = ckpt_info["adapter_path"]
                    aw_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    if not os.path.exists(aw_path):
                        aw_path = os.path.join(adapter_path, "adapter_model.bin")

                    if os.path.exists(aw_path):
                        from safetensors.torch import load_file

                        state_dict = (
                            load_file(aw_path) if aw_path.endswith(".safetensors")
                            else torch.load(aw_path, map_location=self.module.device, weights_only=True)
                        )
                        decoder = self.module.model.decoder
                        if hasattr(decoder, "_forward_module"):
                            decoder = decoder._forward_module
                        decoder.load_state_dict(state_dict, strict=False)

                        start_epoch = ckpt_info["epoch"]
                        global_step = ckpt_info["global_step"]
                        parts = [f"[OK] Resumed from epoch {start_epoch}, step {global_step}"]
                        if ckpt_info["loaded_optimizer"]:
                            parts.append("optimizer OK")
                        if ckpt_info["loaded_scheduler"]:
                            parts.append("scheduler OK")
                        yield TrainingUpdate(0, 0.0, ", ".join(parts), kind="info")
                    else:
                        yield TrainingUpdate(0, 0.0, f"[WARN] Adapter weights not found in {adapter_path}", kind="warn")
                else:
                    yield TrainingUpdate(0, 0.0, f"[WARN] No valid checkpoint in {cfg.resume_from}", kind="warn")
            except Exception as exc:
                logger.exception("Failed to load checkpoint")
                yield TrainingUpdate(0, 0.0, f"[WARN] Checkpoint load failed: {exc} -- starting fresh", kind="warn")
                start_epoch = 0
                global_step = 0

        # -- Training loop --------------------------------------------------
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()

        for epoch in range(start_epoch, cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    # Undo the per-step /G so the yielded number is the true avg raw loss.
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped by user", kind="complete")
                    tb.close()
                    return

                loss = self.module.training_step(batch)
                loss = loss / cfg.gradient_accumulation_steps
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    # Gradient clipping
                    self.fabric.clip_gradients(
                        self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                    )

                    optimizer.step()
                    scheduler.step()

                    global_step += 1

                    # accumulated_loss contains values already divided by G for
                    # gradient scaling.  Multiply back so the displayed metric
                    # reflects the true per-sample loss.
                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step

                    # -- Logging (lightweight) --------------------------------
                    _lr = scheduler.get_last_lr()[0]
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)
                        yield TrainingUpdate(
                            step=global_step,
                            loss=avg_loss,
                            msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step",
                            epoch=epoch + 1,
                            max_epochs=cfg.max_epochs,
                            lr=_lr,
                        )

                    # -- Logging (heavy -- per-layer grad norms) ---------------
                    if global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)

                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush remainder
            if accumulation_step > 0:
                self.fabric.clip_gradients(
                    self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                _lr = scheduler.get_last_lr()[0]
                if global_step % cfg.log_every == 0:
                    tb.log_loss(avg_loss, global_step)
                    tb.log_lr(_lr, global_step)
                    yield TrainingUpdate(
                        step=global_step, loss=avg_loss,
                        msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                    )

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}",
                kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
            )

            # Free fragmented VRAM between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpoint
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}")
                save_training_checkpoint(
                    self.module.model, optimizer, scheduler,
                    epoch + 1, global_step, ckpt_dir,
                )
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[OK] Checkpoint saved at epoch {epoch + 1}",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                )

        # -- Final save -----------------------------------------------------
        final_path = str(output_dir / "final")
        save_lora_weights(self.module.model, final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        tb.flush()
        tb.close()
        yield TrainingUpdate(
            step=global_step, loss=final_loss,
            msg=f"[OK] Training complete! LoRA saved to {final_path}",
            kind="complete",
        )

    # ------------------------------------------------------------------
    # Basic (non-Fabric) fallback
    # ------------------------------------------------------------------

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        yield TrainingUpdate(0, 0.0, "[INFO] Starting basic training loop (no Fabric)", kind="info")

        tb = TrainingLogger(cfg.effective_log_dir)
        train_loader = data_module.train_dataloader()

        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        if not trainable_params:
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            tb.close()
            return

        device_type = self.module.device_type if hasattr(self.module, "device_type") else str(self.module.device).split(":")[0]
        optimizer_type = getattr(cfg, "optimizer_type", "adamw")
        optimizer = build_optimizer(
            trainable_params,
            optimizer_type=optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device_type=device_type,
        )

        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
        total_steps = steps_per_epoch * cfg.max_epochs

        scheduler_type = getattr(cfg, "scheduler_type", "cosine")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            lr=cfg.learning_rate,
            optimizer_type=optimizer_type,
        )

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        self.module.model.decoder.train()

        for epoch in range(cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped", kind="complete")
                    tb.close()
                    return

                loss = self.module.training_step(batch)
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                    _lr = scheduler.get_last_lr()[0]
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)
                        yield TrainingUpdate(
                            step=global_step, loss=avg_loss,
                            msg=f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                        )

                    if global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush remainder
            if accumulation_step > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                _lr = scheduler.get_last_lr()[0]
                if global_step % cfg.log_every == 0:
                    tb.log_loss(avg_loss, global_step)
                    tb.log_lr(_lr, global_step)
                    yield TrainingUpdate(
                        step=global_step, loss=avg_loss,
                        msg=f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                        kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                    )

                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s",
                kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}")
                save_lora_weights(self.module.model, ckpt_dir)
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg="[OK] Checkpoint saved",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                )

        final_path = str(output_dir / "final")
        save_lora_weights(self.module.model, final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        tb.flush()
        tb.close()
        yield TrainingUpdate(
            step=global_step, loss=final_loss,
            msg=f"[OK] Training complete! LoRA saved to {final_path}",
            kind="complete",
        )
