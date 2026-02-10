#!/usr/bin/env bash
# Test Environment Auto-Detection - Linux/macOS
# This script tests the environment detection logic

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "ACE-Step Environment Detection Test"
echo "========================================"
echo

OS_NAME="$(uname)"
ARCH="$(uname -m)"
echo "[Info] Platform: $OS_NAME ($ARCH)"
echo

# Test 1: Check for uv
echo "[Test 1] Checking for uv command..."
if command -v uv &>/dev/null; then
    echo "[PASS] uv detected"
    uv --version
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    echo "[PASS] uv detected at $HOME/.local/bin/uv (not in PATH)"
    "$HOME/.local/bin/uv" --version
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
    echo "[PASS] uv detected at $HOME/.cargo/bin/uv (not in PATH)"
    "$HOME/.cargo/bin/uv" --version
else
    echo "[INFO] uv not found"
    echo "To install uv, run: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
echo

# Test 2: Check project.scripts in pyproject.toml
echo "[Test 2] Checking project scripts..."
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    echo "[PASS] pyproject.toml found"
    echo
    echo "Available scripts:"
    grep -E "^acestep" "$SCRIPT_DIR/pyproject.toml" 2>/dev/null | head -5 || echo "  (not found)"
else
    echo "[FAIL] pyproject.toml not found"
fi
echo

# Test 3: Check accelerator
echo "[Test 3] Checking accelerator..."
if [[ "$OS_NAME" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "[INFO] Apple Silicon - MLX backend available"
    echo "  Recommended: ./start_gradio_ui_macos.sh"
elif [[ "$OS_NAME" == "Linux" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        echo "[INFO] NVIDIA CUDA GPU available"
        echo "  Recommended: ./start_gradio_ui.sh"
    else
        echo "[INFO] No NVIDIA GPU detected - CPU mode"
        echo "  Recommended: ./start_gradio_ui.sh"
    fi
else
    echo "[INFO] Platform: $OS_NAME $ARCH"
fi
echo

# Test 4: Environment selection logic
echo "[Test 4] Environment selection logic..."
if command -v uv &>/dev/null || [[ -x "$HOME/.local/bin/uv" ]] || [[ -x "$HOME/.cargo/bin/uv" ]]; then
    echo "[RESULT] Will use: uv package manager"
    echo "Command: uv run acestep"
else
    echo "[ERROR] uv not found!"
    echo "Please install uv first."
fi
echo

echo "========================================"
echo "Test Complete"
echo "========================================"
