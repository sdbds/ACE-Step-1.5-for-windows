"""
Side-Step TUI - Full Interactive Terminal User Interface

A Textual-based TUI for ACE-Step LoRA training with live monitoring,
interactive configuration, and dataset management.
"""

from __future__ import annotations

__all__ = ["SideStepApp", "run_tui"]


def run_tui() -> None:
    """Launch the Side-Step TUI application."""
    from acestep.training_v2.tui.app import SideStepApp
    
    app = SideStepApp()
    app.run()
