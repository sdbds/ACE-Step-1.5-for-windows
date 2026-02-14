"""
ACE-Step Training V2 -- Corrected LoRA Fine-Tuning CLI

Non-destructive parallel module providing corrected training procedures
that match each model variant's own forward() training logic.

Subcommands:
    vanilla   -- Reproduce existing (bugged) training for backward compatibility
    fixed     -- Corrected training: continuous timesteps + CFG dropout
    selective -- Fixed + dataset-specific layer/module selection
    estimate  -- Gradient sensitivity analysis (no training)
    compare-configs -- Compare module configs across datasets
"""

__version__ = "2.0.0"
