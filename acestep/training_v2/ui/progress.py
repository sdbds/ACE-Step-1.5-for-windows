"""
Live training progress display using Rich.

Renders a live-updating dashboard that shows:
    - Epoch progress bar with ETA
    - Current metrics (loss, learning rate, speed)
    - GPU VRAM usage bar
    - Scrolling log of recent messages

Falls back to plain ``print(msg)`` when Rich is unavailable or stdout
is not a TTY.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional, Tuple, Union

from acestep.training_v2.ui import TrainingUpdate, console, is_rich_active
from acestep.training_v2.ui.gpu_monitor import GPUMonitor


# ---- Training statistics tracker --------------------------------------------

@dataclass
class TrainingStats:
    """Accumulates statistics during training for the live display and
    the post-training summary.
    """

    start_time: float = 0.0
    first_loss: float = 0.0
    best_loss: float = float("inf")
    last_loss: float = 0.0
    last_lr: float = 0.0
    _lr_seen: bool = False
    current_epoch: int = 0
    max_epochs: int = 0
    current_step: int = 0
    total_steps_estimate: int = 0
    steps_this_session: int = 0
    peak_vram_mb: float = 0.0
    last_epoch_time: float = 0.0
    _step_times: list = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        return _fmt_duration(self.elapsed)

    @property
    def samples_per_sec(self) -> float:
        if not self._step_times or len(self._step_times) < 2:
            return 0.0
        dt = self._step_times[-1] - self._step_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._step_times) - 1) / dt

    @property
    def eta_seconds(self) -> float:
        if self.max_epochs <= 0 or self.current_epoch <= 0:
            return 0.0
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0
        progress = self.current_epoch / self.max_epochs
        if progress <= 0:
            return 0.0
        return elapsed * (1.0 / progress - 1.0)

    @property
    def eta_str(self) -> str:
        eta = self.eta_seconds
        if eta <= 0:
            return "--"
        return _fmt_duration(eta)

    def record_step(self) -> None:
        now = time.time()
        self._step_times.append(now)
        # Keep a sliding window of 50 timestamps for speed calculation
        if len(self._step_times) > 50:
            self._step_times = self._step_times[-50:]


def _fmt_duration(seconds: float) -> str:
    """Format seconds to ``1h 23m 45s`` or ``12m 34s`` or ``45s``."""
    if seconds < 0:
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---- Rich live display builder ----------------------------------------------

def _build_display(
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
) -> "Rich renderable":
    """Build the composite Rich renderable for one Live refresh."""
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    # -- Epoch progress -------------------------------------------------------
    epoch_pct = 0.0
    if stats.max_epochs > 0:
        epoch_pct = stats.current_epoch / stats.max_epochs
    progress_bar = ProgressBar(total=100, completed=int(epoch_pct * 100), width=40)

    epoch_line = Text()
    epoch_line.append("  Epoch ", style="dim")
    epoch_line.append(f"{stats.current_epoch}", style="bold")
    epoch_line.append(f" / {stats.max_epochs}  ", style="dim")
    epoch_line.append_text(Text.from_markup(f"  Step {stats.current_step}"))
    epoch_line.append(f"  |  ETA {stats.eta_str}", style="dim")

    # -- Metrics table --------------------------------------------------------
    metrics = Table(show_header=False, show_edge=False, pad_edge=False, box=None, expand=True)
    metrics.add_column("key", style="dim", ratio=1)
    metrics.add_column("val", ratio=1)
    metrics.add_column("key2", style="dim", ratio=1)
    metrics.add_column("val2", ratio=1)

    # Loss formatting: color-code direction
    loss_str = f"{stats.last_loss:.4f}" if stats.last_loss > 0 else "--"
    best_str = f"{stats.best_loss:.4f}" if stats.best_loss < float("inf") else "--"
    lr_str = f"{stats.last_lr:.2e}" if stats._lr_seen else "--"
    speed_str = f"{stats.samples_per_sec:.1f} steps/s" if stats.samples_per_sec > 0 else "--"

    metrics.add_row("Loss", f"[bold]{loss_str}[/]", "Best", f"[green]{best_str}[/]")
    metrics.add_row("LR", lr_str, "Speed", speed_str)
    metrics.add_row("Elapsed", stats.elapsed_str, "Epoch time", f"{stats.last_epoch_time:.1f}s" if stats.last_epoch_time > 0 else "--")

    # -- VRAM bar -------------------------------------------------------------
    if gpu.available:
        snap = gpu.snapshot()
        pct = snap.percent
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar_color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
        bar = f"[{bar_color}]{'#' * filled}[/][dim]{'-' * (bar_width - filled)}[/]"
        vram_line = (
            f"  VRAM {bar}  "
            f"{snap.used_gb:.1f} / {snap.total_gb:.1f} GiB  "
            f"[dim]({pct:.0f}%)[/]"
        )
    else:
        vram_line = "  [dim]VRAM monitoring not available[/]"

    # -- Recent log -----------------------------------------------------------
    log_text = Text()
    for msg in recent_msgs[-5:]:
        if msg.startswith("[OK]"):
            log_text.append(f"  {msg}\n", style="green")
        elif msg.startswith("[WARN]"):
            log_text.append(f"  {msg}\n", style="yellow")
        elif msg.startswith("[FAIL]"):
            log_text.append(f"  {msg}\n", style="red")
        elif msg.startswith("[INFO]"):
            log_text.append(f"  {msg}\n", style="blue")
        else:
            log_text.append(f"  {msg}\n", style="dim")

    # -- Assemble panel -------------------------------------------------------
    from rich.console import Group

    parts = [
        epoch_line,
        Text(""),
        Text.from_markup(f"  {progress_bar}  [dim]{epoch_pct * 100:.0f}%[/]"),
        Text(""),
        metrics,
        Text(""),
        Text.from_markup(vram_line),
        Text(""),
        log_text,
    ]

    return Panel(
        Group(*parts),
        title="[bold]Side-Step Training Progress[/]",
        border_style="green",
        padding=(0, 1),
    )


# ---- Main entry point -------------------------------------------------------

def track_training(
    training_iter: Iterator[Union[Tuple[int, float, str], TrainingUpdate]],
    max_epochs: int,
    device: str = "cuda:0",
    refresh_per_second: int = 2,
) -> TrainingStats:
    """Consume training yields and display live progress.

    Args:
        training_iter: Generator yielding ``(step, loss, msg)`` or
            ``TrainingUpdate`` objects.
        max_epochs: Total number of epochs (for progress bar).
        device: Device string for GPU monitoring.
        refresh_per_second: Rich Live refresh rate.

    Returns:
        Final ``TrainingStats`` for the summary display.
    """
    stats = TrainingStats(start_time=time.time(), max_epochs=max_epochs)
    gpu = GPUMonitor(device=device, interval=3.0)
    recent_msgs: list[str] = []

    if is_rich_active() and console is not None:
        return _track_rich(training_iter, stats, gpu, recent_msgs, refresh_per_second)
    else:
        return _track_plain(training_iter, stats, gpu, recent_msgs)


def _track_rich(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
    refresh_per_second: int,
) -> TrainingStats:
    from rich.live import Live

    assert console is not None

    with Live(
        _build_display(stats, gpu, recent_msgs),
        console=console,
        refresh_per_second=refresh_per_second,
        transient=True,  # Clear the live display when done
    ) as live:
        for update in training_iter:
            # Unpack (works for both tuples and TrainingUpdate)
            if isinstance(update, TrainingUpdate):
                step, loss, msg = update.step, update.loss, update.msg
                _process_structured(update, stats)
            else:
                step, loss, msg = update
                _process_tuple(step, loss, msg, stats)

            recent_msgs.append(msg)
            if len(recent_msgs) > 20:
                recent_msgs.pop(0)

            live.update(_build_display(stats, gpu, recent_msgs))

    # Record peak VRAM
    stats.peak_vram_mb = gpu.peak_mb()
    return stats


def _track_plain(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
) -> TrainingStats:
    for update in training_iter:
        if isinstance(update, TrainingUpdate):
            step, loss, msg = update.step, update.loss, update.msg
            _process_structured(update, stats)
        else:
            step, loss, msg = update
            _process_tuple(step, loss, msg, stats)

        print(msg)

    stats.peak_vram_mb = gpu.peak_mb()
    return stats


# ---- Update processing helpers ----------------------------------------------

def _process_structured(update: TrainingUpdate, stats: TrainingStats) -> None:
    """Extract stats from a TrainingUpdate."""
    stats.current_step = update.step
    stats.last_loss = update.loss
    stats.current_epoch = update.epoch
    if update.max_epochs > 0:
        stats.max_epochs = update.max_epochs
    if update.lr >= 0 and update.kind == "step":
        stats.last_lr = update.lr
        stats._lr_seen = True
    if update.epoch_time > 0:
        stats.last_epoch_time = update.epoch_time

    if stats.first_loss == 0.0 and update.loss > 0:
        stats.first_loss = update.loss
    if update.loss > 0 and update.loss < stats.best_loss:
        stats.best_loss = update.loss

    if update.kind == "step":
        stats.record_step()
        stats.steps_this_session += 1


def _process_tuple(step: int, loss: float, msg: str, stats: TrainingStats) -> None:
    """Extract stats from a raw ``(step, loss, msg)`` tuple by parsing the msg."""
    stats.current_step = step
    stats.last_loss = loss

    if stats.first_loss == 0.0 and loss > 0:
        stats.first_loss = loss
    if loss > 0 and loss < stats.best_loss:
        stats.best_loss = loss

    # Parse epoch from message patterns:
    #   "Epoch 15/100, Step 450, Loss: 0.7234"
    #   "[OK] Epoch 15/100 in 23.4s, Loss: 0.7234"
    msg_lower = msg.lower()
    if "epoch" in msg_lower:
        try:
            # Find "Epoch X/Y" pattern
            idx = msg.lower().index("epoch")
            rest = msg[idx + 5:].strip()
            parts = rest.split("/")
            if len(parts) >= 2:
                epoch_num = int(parts[0].strip())
                max_part = parts[1].split(",")[0].split(" ")[0].strip()
                max_epochs = int(max_part)
                stats.current_epoch = epoch_num
                if max_epochs > 0:
                    stats.max_epochs = max_epochs
        except (ValueError, IndexError):
            pass

    # Parse epoch time from "[OK] Epoch X/Y in Z.Zs"
    if " in " in msg and ("s," in msg or msg.rstrip().endswith("s")):
        try:
            time_part = msg.split(" in ")[1].split("s")[0].strip()
            stats.last_epoch_time = float(time_part)
        except (IndexError, ValueError):
            pass

    # Detect step messages vs epoch messages for speed tracking
    if msg.startswith("Epoch") and "Step" in msg and "Loss" in msg:
        stats.record_step()
        stats.steps_this_session += 1
