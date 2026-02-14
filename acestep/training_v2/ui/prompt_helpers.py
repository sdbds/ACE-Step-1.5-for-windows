"""
Reusable Rich/fallback prompt helpers for the interactive wizard.

Provides menu selection, typed value prompts, path prompts, boolean prompts,
and section headers -- with automatic Rich fallback to plain ``input()``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from acestep.training_v2.ui import console, is_rich_active

# Windows uses spawn-based multiprocessing which breaks DataLoader workers
IS_WINDOWS = sys.platform == "win32"
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 4


# ---- Helpers ----------------------------------------------------------------

def menu(
    title: str,
    options: list[tuple[str, str]],
    default: int = 1,
) -> str:
    """Display a numbered menu and return the chosen key.

    Args:
        title: Prompt text.
        options: List of ``(key, label)`` tuples.
        default: 1-based default index.

    Returns:
        The ``key`` of the chosen option.
    """
    if is_rich_active() and console is not None:
        console.print()
        console.print(f"  [bold]{title}[/]\n")
        for i, (key, label) in enumerate(options, 1):
            marker = "[bold cyan]>[/]" if i == default else " "
            tag = "  [dim](default)[/]" if i == default else ""
            console.print(f"    {marker} [bold]{i}[/]. {label}{tag}")
        console.print()

        from rich.prompt import IntPrompt
        while True:
            choice = IntPrompt.ask(
                "  Choice",
                default=default,
                console=console,
            )
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
            console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")
    else:
        print(f"\n  {title}\n")
        for i, (key, label) in enumerate(options, 1):
            tag = " (default)" if i == default else ""
            print(f"    {i}. {label}{tag}")
        print()
        while True:
            try:
                raw = input(f"  Choice [{default}]: ").strip()
                choice = int(raw) if raw else default
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"  Please enter a number between 1 and {len(options)}")
            except ValueError:
                print(f"  Please enter a number between 1 and {len(options)}")


def ask(
    label: str,
    default: Any = None,
    required: bool = False,
    type_fn: type = str,
    choices: Optional[list] = None,
) -> Any:
    """Ask for a single value with an optional default.

    Args:
        label: Prompt text.
        default: Default value (None = required).
        required: If True, empty input is rejected.
        type_fn: Cast function (str, int, float).
        choices: Optional list of valid string values.

    Returns:
        The user's input, cast to ``type_fn``.
    """
    if choices:
        choice_str = f" ({'/'.join(str(c) for c in choices)})"
    else:
        choice_str = ""

    if is_rich_active() and console is not None:
        from rich.prompt import Prompt, IntPrompt, FloatPrompt

        prompt_cls = Prompt
        if type_fn is int:
            prompt_cls = IntPrompt
        elif type_fn is float:
            prompt_cls = FloatPrompt

        while True:
            result = prompt_cls.ask(
                f"  {label}{choice_str}",
                default=default if default is not None else ...,
                console=console,
            )
            if result is ...:
                if required:
                    console.print("  [red]This field is required[/]")
                    continue
                return None  # optional field, user pressed Enter to skip
            if required and not str(result).strip():
                console.print("  [red]This field is required[/]")
                continue
            if choices and str(result) not in [str(c) for c in choices]:
                console.print(f"  [red]Must be one of: {', '.join(str(c) for c in choices)}[/]")
                continue
            return type_fn(result) if not isinstance(result, type_fn) else result
    else:
        default_str = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"  {label}{choice_str}{default_str}: ").strip()
            if not raw and default is not None:
                return default
            if not raw and required:
                print("  This field is required")
                continue
            try:
                val = type_fn(raw)
                if choices and str(val) not in [str(c) for c in choices]:
                    print(f"  Must be one of: {', '.join(str(c) for c in choices)}")
                    continue
                return val
            except (ValueError, TypeError):
                print(f"  Invalid input, expected {type_fn.__name__}")


def ask_path(
    label: str,
    default: Optional[str] = None,
    must_exist: bool = False,
) -> str:
    """Ask for a filesystem path, optionally validating existence."""
    while True:
        val = ask(label, default=default, required=True)
        if must_exist and not Path(val).exists():
            if is_rich_active() and console is not None:
                console.print(f"  [red]Path not found: {val}[/]")
            else:
                print(f"  Path not found: {val}")
            continue
        return val


def ask_bool(label: str, default: bool = True) -> bool:
    """Ask for a yes/no boolean value."""
    choices = ["yes", "no"]
    default_str = "yes" if default else "no"
    result = ask(label, default=default_str, choices=choices)
    return result.lower() in ("yes", "y", "true", "1")


def section(title: str) -> None:
    """Print a section header."""
    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]--- {title} ---[/]\n")
    else:
        print(f"\n  --- {title} ---\n")
