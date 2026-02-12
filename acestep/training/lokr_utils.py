"""
LoKR Utilities for ACE-Step (via LyCORIS)

Provides utilities for injecting LoKR (Low-Rank Kronecker Product) adapters
into the DiT decoder model using the lycoris-lora library.
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
from loguru import logger

import torch

try:
    from lycoris import create_lycoris, LycorisNetwork
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    logger.warning("LyCORIS library not installed. LoKR training will not be available. "
                    "Install with: pip install lycoris-lora")

from acestep.training.configs import LoKRConfig


def check_lycoris_available() -> bool:
    """Check if LyCORIS library is available."""
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
            "LyCORIS library is required for LoKR training. "
            "Install with: pip install lycoris-lora"
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

    # Freeze all non-LoKR parameters BEFORE injection so newly created LoKR params
    # are not accidentally frozen.
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Apply preset to filter target modules
    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )

    # Create LyCORIS network with LoKR algorithm
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
        # DoRA mode: set via kwargs if supported
        try:
            lycoris_net2 = create_lycoris(
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
            # If successful, restore and use the DoRA version
            lycoris_net = lycoris_net2
        except Exception as e:
            logger.warning(f"DoRA (weight_decompose) not supported in this LyCORIS version: {e}")

    # Apply the LoKR wrappers to the decoder
    lycoris_net.apply_to()

    # Register LyCORIS network on the decoder so its parameters are discoverable
    # via decoder/model .parameters() traversal (optimizer/clipping/statistics).
    decoder._lycoris_net = lycoris_net

    # Enable gradients for LoKR parameters.
    # LyCORIS may not expose all trainable params via lycoris_net.parameters(),
    # but lycoris_net.loras contains the injected modules.
    lokr_param_list = []
    for m in getattr(lycoris_net, "loras", []) or []:
        for p in m.parameters():
            p.requires_grad = True
            lokr_param_list.append(p)
    if not lokr_param_list:
        for p in lycoris_net.parameters():
            p.requires_grad = True
            lokr_param_list.append(p)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    # De-dup in case params are shared / yielded multiple times
    _uniq = {}
    for p in lokr_param_list:
        _uniq[id(p)] = p
    lokr_params = sum(p.numel() for p in _uniq.values())
    trainable_params = sum(p.numel() for p in _uniq.values() if p.requires_grad)

    info = {
        "total_params": total_params,
        "lokr_params": lokr_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
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

    return model, lycoris_net, info


def save_lokr_weights(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Save LoKR adapter weights.

    Args:
        lycoris_net: The LyCORIS network wrapper
        output_dir: Directory to save weights
        dtype: Optional dtype to save in (e.g. torch.float16 for smaller files)
        metadata: Optional metadata dict to include in safetensors

    Returns:
        Path to saved weights file
    """
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "lokr_weights.safetensors")

    save_metadata = {"algo": "lokr", "format": "lycoris"}
    if metadata:
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, str):
                save_metadata[k] = v
            else:
                save_metadata[k] = json.dumps(v, ensure_ascii=False)

    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)
    logger.info(f"LoKR weights saved to {weights_path}")

    return weights_path


def load_lokr_weights(
    lycoris_net: "LycorisNetwork",
    weights_path: str,
) -> Dict[str, Any]:
    """Load LoKR adapter weights into an existing LyCORIS network.

    Args:
        lycoris_net: The LyCORIS network wrapper (must already be applied)
        weights_path: Path to saved weights (.safetensors or .pt)

    Returns:
        Dictionary with load info (missing_keys, unexpected_keys)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoKR weights not found: {weights_path}")

    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoKR weights loaded from {weights_path}")

    return result


def save_lokr_training_checkpoint(
    lycoris_net: "LycorisNetwork",
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config: Optional[LoKRConfig] = None,
) -> str:
    """Save a training checkpoint including LoKR weights and training state.

    Args:
        lycoris_net: The LyCORIS network wrapper
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        global_step: Current global step
        output_dir: Directory to save checkpoint
        lokr_config: Optional LoKR config to save alongside

    Returns:
        Path to saved checkpoint directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save LoKR weights
    metadata = None
    if lokr_config is not None:
        metadata = {"lokr_config": lokr_config.to_dict()}
    save_lokr_weights(lycoris_net, output_dir, metadata=metadata)

    # Save training state
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if lokr_config is not None:
        training_state["lokr_config"] = lokr_config.to_dict()

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(training_state, state_path)

    logger.info(f"LoKR training checkpoint saved to {output_dir} (epoch {epoch}, step {global_step})")
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
