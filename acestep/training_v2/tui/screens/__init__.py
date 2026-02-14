"""
Side-Step TUI Screens

Each screen represents a major view in the application:
- Dashboard: Main menu and quick stats
- TrainingConfig: Form for configuring training runs
- TrainingMonitor: Live view of running training
- DatasetBrowser: Browse and preprocess datasets
- PreprocessMonitor: Live preprocessing progress
- RunHistory: View past runs and checkpoints
- Settings: Application preferences
- EstimateConfig: Gradient sensitivity configuration
- EstimateMonitor: Estimation progress and results
"""

from __future__ import annotations

__all__ = [
    "DashboardScreen",
    "TrainingConfigScreen", 
    "TrainingMonitorScreen",
    "DatasetBrowserScreen",
    "PreprocessMonitorScreen",
    "RunHistoryScreen",
    "SettingsScreen",
    "EstimateConfigScreen",
    "EstimateMonitorScreen",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "DashboardScreen":
        from acestep.training_v2.tui.screens.dashboard import DashboardScreen
        return DashboardScreen
    elif name == "TrainingConfigScreen":
        from acestep.training_v2.tui.screens.training_config import TrainingConfigScreen
        return TrainingConfigScreen
    elif name == "TrainingMonitorScreen":
        from acestep.training_v2.tui.screens.training_monitor import TrainingMonitorScreen
        return TrainingMonitorScreen
    elif name == "DatasetBrowserScreen":
        from acestep.training_v2.tui.screens.dataset_browser import DatasetBrowserScreen
        return DatasetBrowserScreen
    elif name == "PreprocessMonitorScreen":
        from acestep.training_v2.tui.screens.preprocess_monitor import PreprocessMonitorScreen
        return PreprocessMonitorScreen
    elif name == "RunHistoryScreen":
        from acestep.training_v2.tui.screens.run_history import RunHistoryScreen
        return RunHistoryScreen
    elif name == "SettingsScreen":
        from acestep.training_v2.tui.screens.settings import SettingsScreen
        return SettingsScreen
    elif name == "EstimateConfigScreen":
        from acestep.training_v2.tui.screens.estimate import EstimateConfigScreen
        return EstimateConfigScreen
    elif name == "EstimateMonitorScreen":
        from acestep.training_v2.tui.screens.estimate import EstimateMonitorScreen
        return EstimateMonitorScreen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
