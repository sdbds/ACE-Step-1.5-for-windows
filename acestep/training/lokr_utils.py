"""LoKr utilities for ACE-Step training and inference."""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from acestep.training.configs import LoKRConfig
from acestep.training.path_safety import safe_path

try:
    from lycoris import LycorisNetwork, create_lycoris

    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    LycorisNetwork = Any  # type: ignore[assignment,misc]
    logger.warning(
        "LyCORIS library not installed. LoKr training/inference unavailable. "
        "Install with: pip install lycoris-lora"
    )


def check_lycoris_available() -> bool:
    """Check if LyCORIS is importable."""
    return LYCORIS_AVAILABLE


def inject_lokr_into_dit(
    model,
    lokr_config: LoKRConfig,
    multiplier: float = 1.0,
) -> Tuple[Any, "LycorisNetwork", Dict[str, Any]]:

    """Inject LoKR adapters into the DiT decoder of the model using LyCORIS.

    Args:
        model: The AceStepConditionGenerationModel
        lokr_config: LoKR configuration
        multiplier: LoKR output multiplier (default 1.0)

    Returns:
        Tuple of (model, lycoris_network, info_dict)
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoKr training. Install with: pip install lycoris-lora"
        )

    decoder = model.decoder

    prev_net = getattr(decoder, "_lycoris_net", None)
    if prev_net is not None:
        try:
            if hasattr(prev_net, "restore"):
                prev_net.restore()
        except Exception:
            pass
        try:
            delattr(decoder, "_lycoris_net")
        except Exception:
            pass

    for _, param in model.named_parameters():
        param.requires_grad = False

    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )

    lycoris_net = create_lycoris(
        decoder,
        multiplier,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                decoder,
                multiplier,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net

    # IMPORTANT: LyCORIS preset/create_lycoris already handles target-module
    # selection. Re-filtering by fragile name matching here can accidentally
    # freeze valid LoKr tensors (e.g. lokr_w2_b), causing silent quality
    # regression with many all-zero saved tensors.
    lokr_param_list = []
    for module in getattr(lycoris_net, "loras", []) or []:
        for param in module.parameters():
            param.requires_grad = True
            lokr_param_list.append(param)

    if not lokr_param_list:
        for param in lycoris_net.parameters():
            param.requires_grad = True
            lokr_param_list.append(param)

    unique_params = {id(p): p for p in lokr_param_list}
    total_params = sum(p.numel() for p in model.parameters())
    lokr_params = sum(p.numel() for p in unique_params.values())
    trainable_params = sum(p.numel() for p in unique_params.values() if p.requires_grad)

    info = {
        "total_params": total_params,
        "lokr_params": lokr_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
        "linear_dim": lokr_config.linear_dim,
        "linear_alpha": lokr_config.linear_alpha,
        "factor": lokr_config.factor,
        "algo": "lokr",
        "target_modules": lokr_config.target_modules,
    }


    logger.info("LoKR injected into DiT decoder:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  LoKR parameters: {lokr_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    logger.info(f"  linear_dim: {lokr_config.linear_dim}, linear_alpha: {lokr_config.linear_alpha}")
    logger.info(f"  factor: {lokr_config.factor}, decompose_both: {lokr_config.decompose_both}")

    logger.info("LoKr injected into decoder")
    logger.info(
        f"LoKr trainable params: {trainable_params:,}/{total_params:,} "
        f"({info['trainable_ratio']:.2%})"
    )

    return model, lycoris_net, info


def save_lokr_weights(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Save LoKr weights to safetensors."""
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "lokr_weights.safetensors")

    save_metadata: Dict[str, str] = {"algo": "lokr", "format": "lycoris"}
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str):
                save_metadata[key] = value
            else:
                save_metadata[key] = json.dumps(value, ensure_ascii=True)

    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)
    logger.info(f"LoKr weights saved to {weights_path}")
    return weights_path


def load_lokr_weights(lycoris_net: "LycorisNetwork", weights_path: str) -> Dict[str, Any]:
    """Load LoKr weights into an injected LyCORIS network."""
    weights_path = safe_path(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoKr weights not found: {weights_path}")
    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoKr weights loaded from {weights_path}")
    return result


def save_lokr_training_checkpoint(
    lycoris_net: "LycorisNetwork",
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config: Optional[LoKRConfig] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save LoKr weights plus optimizer/scheduler state."""
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata: Dict[str, Any] = {}
    if lokr_config is not None:
        metadata["lokr_config"] = lokr_config.to_dict()
    if run_metadata is not None:
        metadata["run_metadata"] = run_metadata
    metadata = metadata or None
    save_lokr_weights(lycoris_net, output_dir, metadata=metadata)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if lokr_config is not None:
        state["lokr_config"] = lokr_config.to_dict()
    if run_metadata is not None:
        state["run_metadata"] = run_metadata

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(state, state_path)

    logger.info(f"LoKr checkpoint saved to {output_dir} (epoch={epoch}, step={global_step})")
    return output_dir


def load_lokr_training_checkpoint(
    checkpoint_dir: str,
    lycoris_net: Optional["LycorisNetwork"] = None,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load LoKR training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        lycoris_net: Optional LyCORIS network to load weights into
        optimizer: Optimizer instance to load state into (optional)
        scheduler: Scheduler instance to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint info
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "weights_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
        "lokr_config": None,
    }

    # Find weights file
    weights_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "lokr_weights.pt")
    if os.path.exists(weights_path):
        result["weights_path"] = weights_path
        if lycoris_net is not None:
            load_lokr_weights(lycoris_net, weights_path)

    # Load training state
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        map_location = device if device else "cpu"
        training_state = torch.load(state_path, map_location=map_location)

        result["epoch"] = training_state.get("epoch", 0)
        result["global_step"] = training_state.get("global_step", 0)
        result["lokr_config"] = training_state.get("lokr_config", None)

        if optimizer is not None and "optimizer_state_dict" in training_state:
            try:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                result["loaded_optimizer"] = True
                logger.info("Optimizer state loaded from LoKR checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        if scheduler is not None and "scheduler_state_dict" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True
                logger.info("Scheduler state loaded from LoKR checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Loaded LoKR checkpoint from epoch {result['epoch']}, step {result['global_step']}")
    else:
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_dir)
        if match:
            result["epoch"] = int(match.group(1))

    return result


def restore_lokr(lycoris_net: "LycorisNetwork") -> None:
    """Remove LoKR adapters and restore the original model weights.

    Args:
        lycoris_net: The LyCORIS network wrapper to remove
    """
    if lycoris_net is not None:
        lycoris_net.restore()
        logger.info("LoKR adapters removed, original model restored")


def get_lokr_info(lycoris_net: "LycorisNetwork") -> Dict[str, Any]:
    """Get information about LoKR adapters.

    Args:
        lycoris_net: The LyCORIS network wrapper

    Returns:
        Dictionary with LoKR information
    """
    info = {
        "has_lokr": False,
        "lokr_params": 0,
        "num_modules": 0,
    }

    if lycoris_net is None:
        return info

    lokr_params = sum(p.numel() for p in lycoris_net.parameters())
    num_modules = len(list(lycoris_net.loras))

    info["has_lokr"] = lokr_params > 0
    info["lokr_params"] = lokr_params
    info["num_modules"] = num_modules

    return info
