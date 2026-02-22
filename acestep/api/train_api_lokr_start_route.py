"""LoKR training start route registration."""

from __future__ import annotations

import os
import re
import threading
import time
import math
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from acestep.api.train_api_models import StartLoKRTrainingRequest, initialize_training_state
from acestep.api.train_api_runtime import RuntimeComponentManager, unwrap_module
from acestep.handler import AceStepHandler


def _format_duration(seconds: Optional[float]) -> str:
    """Format seconds into H:MM:SS for status messages."""
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _estimate_total_steps(
    num_samples: int,
    batch_size: int,
    gradient_accumulation: int,
    max_epochs: int,
) -> int:
    """Estimate optimizer steps using the same rounding rules as trainer loops."""
    micro_batches_per_epoch = max(1, math.ceil(max(1, num_samples) / max(1, batch_size)))
    steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / max(1, gradient_accumulation)))
    return steps_per_epoch * max(1, max_epochs)


def _decorate_status_line(
    base_status: str,
    elapsed_seconds: float,
    steps_per_second: float,
    eta_seconds: Optional[float],
) -> str:
    """Append elapsed time, speed, and ETA to a status line."""
    elapsed_text = _format_duration(elapsed_seconds)
    speed_text = f"{steps_per_second:.2f} step/s" if steps_per_second > 0 else "-- step/s"
    eta_text = _format_duration(eta_seconds)
    return f"{base_status} | elapsed {elapsed_text} | {speed_text} | ETA {eta_text}"


def _resolve_lokr_dataloader_config(device: Any) -> Dict[str, Any]:
    """Resolve DataLoader tuning defaults for LoKR training.

    Args:
        device: Handler device object or string.

    Returns:
        Mapping with num_workers/pin_memory/prefetch_factor/persistent_workers/pin_memory_device.
    """
    if hasattr(device, "type"):
        device_type = str(device.type).lower()
    else:
        device_type = str(device).split(":", 1)[0].lower()

    if device_type == "cuda":
        cfg = {
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory_device": "cuda",
        }
    elif device_type == "xpu":
        cfg = {
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory_device": "",
        }
    else:
        cfg = {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "pin_memory_device": "",
        }

    return cfg


def register_lokr_training_start_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    start_tensorboard: Callable[[FastAPI, str], Optional[str]],
) -> None:
    """Register the `/v1/training/start_lokr` route."""

    @app.post("/v1/training/start_lokr")
    async def start_lokr_training(request: StartLoKRTrainingRequest, _: None = Depends(verify_api_key)):
        """Start LoKr training from preprocessed tensors."""
        logger.info(f"[LoKR start] tensor_dir={request.tensor_dir!r}, output_dir={request.output_dir!r}")

        initialize_training_state(app)
        training_state = app.state.training_state
        if training_state.get("is_training", False):
            raise HTTPException(status_code=400, detail="Training already in progress")

        handler: AceStepHandler = app.state.handler
        if handler is None or handler.model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        if not hasattr(handler.model, "decoder") or handler.model.decoder is None:
            raise HTTPException(
                status_code=500,
                detail="Decoder not found. Please reload the model via /v1/reinitialize before training.",
            )

        handler.model.decoder = unwrap_module(handler.model.decoder)
        mgr = RuntimeComponentManager(handler=handler, llm=app.state.llm_handler, app_state=app.state)
        mgr.move_decoder_to(str(handler.device))
        mgr.offload_vae_to_cpu()
        mgr.offload_text_encoder_to_cpu()
        mgr.offload_model_encoder_to_cpu()
        mgr.offload_model_tokenizer_to_cpu()
        mgr.offload_model_detokenizer_to_cpu()
        mgr.unload_llm()
        mgr.flush_gpu_cache()

        try:
            from acestep.training.configs import LoKRConfig as LoKRConfigClass, TrainingConfig
            from acestep.training.data_module import build_tensor_shards
            from acestep.training.trainer import LoKRTrainer

            dataloader_cfg = _resolve_lokr_dataloader_config(
                device=getattr(handler, "device", ""),
            )

            shard_result: Dict[str, Any] = {
                "created": False,
                "already_sharded": False,
                "num_samples": 0,
                "num_shards": 0,
                "reason": "disabled",
            }
            if request.auto_shard and request.shard_size > 0:
                try:
                    shard_result = build_tensor_shards(
                        request.tensor_dir,
                        shard_size=request.shard_size,
                    )
                    logger.info(
                        f"[LoKR start] shard status: created={shard_result.get('created')} "
                        f"already_sharded={shard_result.get('already_sharded')} "
                        f"samples={shard_result.get('num_samples')} shards={shard_result.get('num_shards')}"
                    )
                except Exception as exc:
                    logger.warning(f"[LoKR start] auto_shard failed, continuing with original tensors: {exc}")

            factor = request.lokr_factor
            if factor != -1:
                factor = int(factor)
                if factor == 0:
                    factor = 1
                factor = min(factor, 8)

            lokr_config = LoKRConfigClass(
                linear_dim=request.lokr_linear_dim,
                linear_alpha=request.lokr_linear_alpha,
                factor=factor,
                decompose_both=request.lokr_decompose_both,
                use_tucker=request.lokr_use_tucker,
                use_scalar=request.lokr_use_scalar,
                weight_decompose=request.lokr_weight_decompose,
            )
            training_config = TrainingConfig(
                shift=request.training_shift,
                learning_rate=request.learning_rate,
                batch_size=request.train_batch_size,
                gradient_accumulation_steps=request.gradient_accumulation,
                max_epochs=request.train_epochs,
                save_every_n_epochs=request.save_every_n_epochs,
                seed=request.training_seed,
                output_dir=request.output_dir,
                gradient_checkpointing=request.gradient_checkpointing,
                network_weights=request.network_weights,
                num_workers=dataloader_cfg["num_workers"],
                pin_memory=dataloader_cfg["pin_memory"],
                prefetch_factor=dataloader_cfg["prefetch_factor"],
                persistent_workers=dataloader_cfg["persistent_workers"],
                pin_memory_device=dataloader_cfg["pin_memory_device"],
                sample_cache_size=request.sample_cache_size,
            )
            trainer = LoKRTrainer(dit_handler=handler, lokr_config=lokr_config, training_config=training_config)
        except Exception as exc:
            training_state["is_training"] = False
            mgr.restore()
            return wrap_response(None, code=500, error=f"Failed to start LoKR training: {exc}")

        tensorboard_logdir = os.path.join(request.output_dir, "logs")
        os.makedirs(tensorboard_logdir, exist_ok=True)

        run_id = str(uuid4())
        training_state.update(
            {
                "is_training": True,
                "should_stop": False,
                "run_id": run_id,
                "trainer": trainer,
                "tensor_dir": request.tensor_dir,
                "tensorboard_logdir": tensorboard_logdir,
                "current_step": 0,
                "total_steps": 0,
                "current_loss": None,
                "status": "Starting...",
                "loss_history": [],
                "training_log": "Starting...",
                "start_time": time.time(),
                "current_epoch": 0,
                "last_step_time": time.time(),
                "steps_per_second": 0.0,
                "estimated_time_remaining": 0.0,
                "error": None,
                "config": {
                    "adapter_type": "lokr",
                    "lokr_linear_dim": request.lokr_linear_dim,
                    "lokr_linear_alpha": request.lokr_linear_alpha,
                    "lokr_factor": request.lokr_factor,
                    "lokr_decompose_both": request.lokr_decompose_both,
                    "lokr_use_tucker": request.lokr_use_tucker,
                    "lokr_use_scalar": request.lokr_use_scalar,
                    "lokr_weight_decompose": request.lokr_weight_decompose,
                    "learning_rate": request.learning_rate,
                    "epochs": request.train_epochs,
                    "batch_size": request.train_batch_size,
                    "gradient_accumulation": request.gradient_accumulation,
                    "sample_cache_size": request.sample_cache_size,
                    "auto_shard": request.auto_shard,
                    "shard_size": request.shard_size,
                    "shard_result": shard_result,
                    "network_weights": request.network_weights,
                },
                "_component_manager": mgr,
            }
        )
        training_state["tensorboard_url"] = start_tensorboard(app, tensorboard_logdir)

        def _runner() -> None:
            local_run_id = run_id
            log_lines: list = []
            total_steps_estimate: Optional[int] = None
            training_state["last_counted_step"] = 0
            last_step_timestamp: Optional[float] = None
            try:
                for step, loss, status in trainer.train_from_preprocessed(request.tensor_dir, training_state):
                    if training_state.get("run_id") != local_run_id:
                        break
                    training_state["current_step"] = step
                    training_state["current_loss"] = loss
                    text = str(status)
                    match = re.search(r"Epoch (\d+)/(\d+)", text)
                    if match:
                        training_state["current_epoch"] = int(match.group(1))
                        total_epochs = int(match.group(2))
                    else:
                        total_epochs = training_state.get("config", {}).get("epochs", 0)
                    now = time.time()
                    start = training_state.get("start_time", now)
                    elapsed = now - start

                    # Update speed only when global step advances.
                    last_counted_step = int(training_state.get("last_counted_step", 0))
                    if step > last_counted_step:
                        step_delta = step - last_counted_step
                        if last_step_timestamp is None:
                            inst_sps = (step / elapsed) if (step > 0 and elapsed > 0) else 0.0
                        else:
                            inst_sps = step_delta / max(now - last_step_timestamp, 0.001)
                        prev_sps = float(training_state.get("steps_per_second", 0.0))
                        training_state["steps_per_second"] = inst_sps if prev_sps <= 0 else (prev_sps * 0.8 + inst_sps * 0.2)
                        training_state["last_counted_step"] = step
                        last_step_timestamp = now
                        training_state["last_step_time"] = now

                    # Prefer step-based ETA once dataset size is known.
                    if total_steps_estimate is None:
                        run_meta = getattr(trainer, "run_metadata", {}) or {}
                        num_samples = int(run_meta.get("num_samples", 0) or 0)
                        if num_samples > 0:
                            total_steps_estimate = _estimate_total_steps(
                                num_samples=num_samples,
                                batch_size=int(request.train_batch_size),
                                gradient_accumulation=int(request.gradient_accumulation),
                                max_epochs=int(request.train_epochs),
                            )
                            training_state["total_steps"] = total_steps_estimate
                            config = training_state.get("config")
                            if isinstance(config, dict):
                                config["total_steps"] = total_steps_estimate

                    sps = float(training_state.get("steps_per_second", 0.0))
                    eta_seconds: Optional[float] = None
                    if total_steps_estimate is not None and sps > 0 and step >= 0:
                        remaining_steps = max(0, total_steps_estimate - step)
                        eta_seconds = remaining_steps / sps
                        training_state["estimated_time_remaining"] = eta_seconds
                    elif step > 0 and elapsed > 0:
                        # Conservative fallback before total steps are known.
                        current_epoch = training_state.get("current_epoch", 0)
                        if total_epochs > 0 and current_epoch > 0:
                            remaining = elapsed * (total_epochs - current_epoch) / current_epoch
                            eta_seconds = remaining
                            training_state["estimated_time_remaining"] = remaining

                    decorated = _decorate_status_line(
                        base_status=text,
                        elapsed_seconds=elapsed,
                        steps_per_second=sps,
                        eta_seconds=eta_seconds,
                    )
                    training_state["status"] = decorated
                    log_lines.append(decorated)
                    training_state["training_log"] = "\n".join(log_lines[-200:])

                    if loss is not None and loss == loss and step > 0:
                        history = training_state.get("loss_history", [])
                        history.append({"step": step, "loss": float(loss)})
                        training_state["loss_history"] = history[-1000:]
                    if training_state.get("should_stop", False):
                        break
            except Exception as exc:
                training_state["error"] = str(exc)
            finally:
                training_state["is_training"] = False
                try:
                    if handler.model is not None and getattr(handler.model, "decoder", None) is not None:
                        handler.model.decoder = unwrap_module(handler.model.decoder)
                        handler.model.decoder.eval()
                except Exception:
                    logger.exception("Failed to restore decoder wrapper state after LoKR training")
                cm = training_state.pop("_component_manager", None)
                if cm is not None:
                    cm.restore()

        threading.Thread(target=_runner, daemon=True).start()
        return wrap_response(
            {
                "message": "LoKR training started",
                "tensor_dir": request.tensor_dir,
                "output_dir": request.output_dir,
                "config": training_state["config"],
            }
        )
