#!/usr/bin/env bash
# Quick Environment Test - Linux/macOS
# This script tests the environment for ACE-Step

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Quick Environment Test"
echo "========================================"
echo

OS_NAME="$(uname)"
ARCH="$(uname -m)"
echo "[Info] OS: $OS_NAME ($ARCH)"
echo

# Test 1: Check Python
echo "[Test 1] Checking Python..."
if command -v python3 &>/dev/null; then
    echo "[PASS] Python3 found"
    python3 --version
elif command -v python &>/dev/null; then
    echo "[PASS] Python found"
    python --version
else
    echo "[INFO] Python not found in PATH"
fi
echo

# Test 2: Check uv
echo "[Test 2] Checking uv..."
UV_FOUND=0
if command -v uv &>/dev/null; then
    echo "[PASS] uv found in PATH"
    uv --version
    UV_FOUND=1
else
    echo "[INFO] uv not found in PATH"
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        echo "[INFO] But uv exists at: $HOME/.local/bin/uv"
        "$HOME/.local/bin/uv" --version
        UV_FOUND=1
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        echo "[INFO] But uv exists at: $HOME/.cargo/bin/uv"
        "$HOME/.cargo/bin/uv" --version
        UV_FOUND=1
    else
        echo "[INFO] uv not installed"
    fi
fi
echo

# Test 3: Check git
echo "[Test 3] Checking git..."
if command -v git &>/dev/null; then
    echo "[PASS] git found"
    git --version
else
    echo "[INFO] git not found"
    if [[ "$OS_NAME" == "Darwin" ]]; then
        echo "Install: xcode-select --install  or  brew install git"
    else
        echo "Install: sudo apt install git  or  sudo yum install git"
    fi
fi
echo

# Test 4: Check GPU / accelerator
echo "[Test 4] Checking GPU/accelerator..."
if [[ "$OS_NAME" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "[PASS] Apple Silicon detected (MPS/MLX available)"
    # Check for MLX
    if command -v python3 &>/dev/null; then
        python3 -c "import mlx; print(f'MLX version: {mlx.__version__}')" 2>/dev/null && echo "[PASS] MLX is available" || echo "[INFO] MLX not installed (will be installed by uv sync)"
    fi
elif [[ "$OS_NAME" == "Linux" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        echo "[PASS] NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (could not query GPU details)"
    else
        echo "[INFO] NVIDIA GPU not detected (nvidia-smi not found)"
        echo "  CUDA is recommended for Linux. CPU mode may be very slow."
    fi
else
    echo "[INFO] No known accelerator detected"
fi
echo

# Test 5: Test internet connectivity
echo "[Test 5] Testing internet connectivity..."
if command -v curl &>/dev/null; then
    if curl -s --connect-timeout 5 -o /dev/null -w "%{http_code}" https://astral.sh 2>/dev/null | grep -q "^[23]"; then
        echo "[PASS] Can access astral.sh"
    else
        echo "[FAIL] Cannot access astral.sh"
    fi
elif command -v wget &>/dev/null; then
    if wget -q --spider --timeout=5 https://astral.sh 2>/dev/null; then
        echo "[PASS] Can access astral.sh"
    else
        echo "[FAIL] Cannot access astral.sh"
    fi
else
    echo "[INFO] Neither curl nor wget found, skipping connectivity test"
fi
echo

# Test 6: Check pyproject.toml
echo "[Test 6] Checking project configuration..."
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    echo "[PASS] pyproject.toml found"
    echo
    echo "Available scripts:"
    grep -E "^(acestep|acestep-api|acestep-download) = " "$SCRIPT_DIR/pyproject.toml" 2>/dev/null || echo "  (scripts section not found)"
else
    echo "[FAIL] pyproject.toml not found"
fi
echo

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo

if [[ $UV_FOUND -eq 1 ]]; then
    echo "[RESULT] Environment: uv"
    echo "  Ready to use!"
    if [[ "$OS_NAME" == "Darwin" && "$ARCH" == "arm64" ]]; then
        echo "  Recommended launcher: ./start_gradio_ui_macos.sh (MLX)"
    else
        echo "  Recommended launcher: ./start_gradio_ui.sh"
    fi
else
    echo "[RESULT] No environment found"
    echo "  Action: Run ./install_uv.sh to install uv"
fi
echo
