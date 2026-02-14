"""
Interactive wizard for ACE-Step Training V2.

Launched when ``python train.py`` is run with no subcommand.  Delegates to
``prompt_helpers`` for Rich/fallback prompts and ``flows`` for each
subcommand's interactive flow.

Split structure:
    prompt_helpers.py  -- reusable prompt utilities (menu, ask, section, ...)
    flows.py           -- wizard flow builders (train, preprocess, estimate)
    wizard.py          -- entrypoint: banner + top-level dispatch (this file)
"""

from __future__ import annotations

import argparse
from typing import Optional

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import menu
from acestep.training_v2.ui.flows import wizard_train, wizard_preprocess, wizard_estimate


def run_wizard() -> Optional[argparse.Namespace]:
    """Launch the interactive wizard.

    Returns:
        A populated ``argparse.Namespace`` ready for the normal dispatch
        logic, or ``None`` if the user chose to exit.
    """
    from acestep.training_v2.ui.banner import show_banner
    show_banner(subcommand="interactive")

    try:
        action = menu(
            "What would you like to do?",
            [
                ("fixed", "Train a LoRA (corrected timesteps + CFG dropout)"),
                ("vanilla", "Train a LoRA (original behavior)"),
                ("preprocess", "Preprocess audio data"),
                ("estimate", "Run gradient estimation"),
                ("exit", "Exit"),
            ],
            default=1,
        )

        if action == "exit":
            return None

        if action == "estimate":
            return wizard_estimate()

        if action == "preprocess":
            return wizard_preprocess()

        # fixed or vanilla
        return wizard_train(mode=action)

    except (KeyboardInterrupt, EOFError):
        if is_rich_active() and console is not None:
            console.print("\n  [dim]Aborted.[/]")
        else:
            print("\n  Aborted.")
        return None
