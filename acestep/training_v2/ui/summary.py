"""
Post-training summary panel.

Displays final statistics after training completes: total time, loss
trajectory, GPU usage, output paths, and next-steps hints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.progress import TrainingStats


def _dir_size_str(path: str) -> str:
    """Return human-readable size of a directory."""
    try:
        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        if total > 1024 ** 3:
            return f"{total / 1024 ** 3:.2f} GiB"
        if total > 1024 ** 2:
            return f"{total / 1024 ** 2:.1f} MiB"
        if total > 1024:
            return f"{total / 1024:.0f} KiB"
        return f"{total} B"
    except Exception:
        return "unknown"


def show_summary(
    stats: TrainingStats,
    output_dir: str,
    log_dir: Optional[str] = None,
) -> None:
    """Display the post-training summary.

    Args:
        stats: Accumulated training statistics from the progress tracker.
        output_dir: Path to the LoRA output directory.
        log_dir: Path to TensorBoard log directory (for the hint).
    """
    if is_rich_active() and console is not None:
        _show_rich(stats, output_dir, log_dir)
    else:
        _show_plain(stats, output_dir, log_dir)


def _show_rich(
    stats: TrainingStats,
    output_dir: str,
    log_dir: Optional[str],
) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    assert console is not None

    # -- Stats table ----------------------------------------------------------
    table = Table(show_header=False, show_edge=False, box=None, pad_edge=True, expand=False)
    table.add_column("key", style="dim", min_width=20)
    table.add_column("val", min_width=30)

    table.add_row("Total time", f"[bold]{stats.elapsed_str}[/]")
    table.add_row("Epochs completed", f"{stats.current_epoch} / {stats.max_epochs}")
    table.add_row("Total steps", str(stats.current_step))

    # Loss trajectory
    if stats.first_loss > 0:
        direction = "down" if stats.last_loss < stats.first_loss else "up"
        color = "green" if direction == "down" else "red"
        pct = abs(stats.last_loss - stats.first_loss) / stats.first_loss * 100
        table.add_row(
            "Loss",
            f"[bold]{stats.first_loss:.4f}[/] -> [{color}]{stats.last_loss:.4f}[/]  "
            f"[dim]({direction} {pct:.1f}%)[/]",
        )
    if stats.best_loss < float("inf"):
        table.add_row("Best loss", f"[green]{stats.best_loss:.4f}[/]")

    # GPU
    if stats.peak_vram_mb > 0:
        table.add_row("Peak VRAM", f"{stats.peak_vram_mb / 1024:.1f} GiB")

    # Speed
    if stats.samples_per_sec > 0:
        table.add_row("Avg speed", f"{stats.samples_per_sec:.1f} steps/s")

    # Output
    table.add_row("", "")  # spacer
    final_dir = Path(output_dir) / "final"
    table.add_row("Output dir", str(output_dir))
    if final_dir.exists():
        table.add_row("LoRA weights", f"{final_dir}  [dim]({_dir_size_str(str(final_dir))})[/]")
    if log_dir:
        table.add_row("TensorBoard", str(log_dir))

    # -- Next steps -----------------------------------------------------------
    hints = Text()
    hints.append("\n  Next steps:\n", style="bold")
    hints.append(f"  1. Use the LoRA:  ", style="dim")
    hints.append(f"load from {final_dir}\n", style="")
    if log_dir:
        hints.append(f"  2. View metrics:  ", style="dim")
        hints.append(f"python launch_tensorboard.py --logdir {log_dir}\n", style="")
    hints.append(f"  3. Generate music with the LoRA via the Gradio UI\n", style="dim")

    # -- Panel ----------------------------------------------------------------
    from rich.console import Group

    console.print(
        Panel(
            Group(table, hints),
            title="[bold green]Training Complete[/]",
            border_style="green",
            padding=(0, 1),
        )
    )


def _show_plain(
    stats: TrainingStats,
    output_dir: str,
    log_dir: Optional[str],
) -> None:
    print("\n" + "=" * 60, file=sys.stderr)
    print("  Training Complete", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Total time .......... {stats.elapsed_str}", file=sys.stderr)
    print(f"  Epochs .............. {stats.current_epoch} / {stats.max_epochs}", file=sys.stderr)
    print(f"  Total steps ......... {stats.current_step}", file=sys.stderr)

    if stats.first_loss > 0:
        print(f"  Loss ................ {stats.first_loss:.4f} -> {stats.last_loss:.4f}", file=sys.stderr)
    if stats.best_loss < float("inf"):
        print(f"  Best loss ........... {stats.best_loss:.4f}", file=sys.stderr)
    if stats.peak_vram_mb > 0:
        print(f"  Peak VRAM ........... {stats.peak_vram_mb / 1024:.1f} GiB", file=sys.stderr)
    if stats.samples_per_sec > 0:
        print(f"  Avg speed ........... {stats.samples_per_sec:.1f} steps/s", file=sys.stderr)

    print(f"\n  Output dir .......... {output_dir}", file=sys.stderr)
    final_dir = Path(output_dir) / "final"
    if final_dir.exists():
        print(f"  LoRA weights ........ {final_dir}  ({_dir_size_str(str(final_dir))})", file=sys.stderr)
    if log_dir:
        print(f"  TensorBoard ......... {log_dir}", file=sys.stderr)

    print(f"\n  Next steps:", file=sys.stderr)
    print(f"    1. Load LoRA from {final_dir}", file=sys.stderr)
    if log_dir:
        print(f"    2. python launch_tensorboard.py --logdir {log_dir}", file=sys.stderr)
    print(f"    3. Generate music with the LoRA via the Gradio UI", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)
