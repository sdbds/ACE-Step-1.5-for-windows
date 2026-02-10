#!/usr/bin/env bash
# ACE-Step REST API Server Launcher - macOS (Apple Silicon / MLX)
# This script launches the REST API server using the MLX backend
# for native Apple Silicon acceleration.
#
# Requirements: macOS with Apple Silicon (M1/M2/M3/M4)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== MLX Configuration ====================
# Force MLX backend for native Apple Silicon acceleration
export ACESTEP_LM_BACKEND="mlx"

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM="false"

# ==================== Server Configuration ====================
HOST="127.0.0.1"
# HOST="0.0.0.0"
PORT=8001

# API key for authentication (optional, value only)
API_KEY=""
# API_KEY="sk-your-secret-key"

# Download source settings (value only: "huggingface" or "modelscope")
DOWNLOAD_SOURCE=""
# DOWNLOAD_SOURCE="huggingface"
# DOWNLOAD_SOURCE="modelscope"

# LLM initialization settings
# Values: auto (default), true (force enable), false (force disable)
export ACESTEP_INIT_LLM=auto
# export ACESTEP_INIT_LLM=true
# export ACESTEP_INIT_LLM=false

# LM model path (value only)
LM_MODEL_PATH=""
# LM_MODEL_PATH="acestep-5Hz-lm-0.6B"

# Update check on startup (set to "false" to disable)
CHECK_UPDATE="true"
# CHECK_UPDATE="false"

# Skip model loading at startup (models will be lazy-loaded on first request)
# Set to "true" to start server quickly without loading models
# export ACESTEP_NO_INIT=false
# export ACESTEP_NO_INIT=true

# ==================== Launch ====================

# Verify macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS only."
    echo "For Linux, use start_api_server.sh instead."
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "WARNING: MLX backend requires Apple Silicon (arm64)."
    echo "Detected architecture: $ARCH"
    echo "Falling back to PyTorch backend."
    unset ACESTEP_LM_BACKEND
fi

# ==================== Startup Update Check ====================
_startup_update_check() {
    [[ "$CHECK_UPDATE" != "true" ]] && return 0
    command -v git &>/dev/null || return 0
    cd "$SCRIPT_DIR" || return 0
    git rev-parse --git-dir &>/dev/null 2>&1 || return 0

    local branch commit remote_commit
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
    commit="$(git rev-parse --short HEAD 2>/dev/null || echo "")"
    [[ -z "$commit" ]] && return 0

    echo "[Update] Checking for updates..."

    # Fetch with timeout (10s) - macOS uses gtimeout from coreutils
    local fetch_ok=0
    if command -v gtimeout &>/dev/null; then
        gtimeout 10 git fetch origin --quiet 2>/dev/null && fetch_ok=1
    elif command -v timeout &>/dev/null; then
        timeout 10 git fetch origin --quiet 2>/dev/null && fetch_ok=1
    else
        git fetch origin --quiet 2>/dev/null && fetch_ok=1
    fi

    if [[ $fetch_ok -eq 0 ]]; then
        echo "[Update] Network unreachable, skipping."
        echo
        return 0
    fi

    remote_commit="$(git rev-parse --short "origin/$branch" 2>/dev/null || echo "")"

    if [[ -z "$remote_commit" || "$commit" == "$remote_commit" ]]; then
        echo "[Update] Already up to date ($commit)."
        echo
        return 0
    fi

    echo
    echo "========================================"
    echo "  Update available!"
    echo "========================================"
    echo "  Current: $commit  ->  Latest: $remote_commit"
    echo
    echo "  Recent changes:"
    git --no-pager log --oneline "HEAD..origin/$branch" 2>/dev/null | head -10
    echo

    read -rp "Update now before starting? (Y/N): " update_choice
    if [[ "${update_choice^^}" == "Y" ]]; then
        if [[ -f "$SCRIPT_DIR/check_update.sh" ]]; then
            bash "$SCRIPT_DIR/check_update.sh"
        else
            echo "Pulling latest changes..."
            git pull --ff-only origin "$branch" 2>/dev/null || {
                echo "[Update] Update failed. Please run: git pull"
            }
        fi
    else
        echo "[Update] Skipped. Run ./check_update.sh to update later."
    fi
    echo
}
_startup_update_check

echo "============================================"
echo "  ACE-Step 1.5 API - macOS Apple Silicon (MLX)"
echo "============================================"
echo
echo "API will be available at: http://${HOST}:${PORT}"
echo "API Documentation: http://${HOST}:${PORT}/docs"
echo

# Check if uv is installed
if ! command -v uv &>/dev/null; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

if ! command -v uv &>/dev/null; then
    echo
    echo "========================================"
    echo "uv package manager not found!"
    echo "========================================"
    echo
    echo "ACE-Step requires the uv package manager."
    echo
    read -rp "Install uv now? (Y/N): " INSTALL_UV

    if [[ "${INSTALL_UV^^}" == "Y" ]]; then
        echo
        bash "$SCRIPT_DIR/install_uv.sh" --silent
        INSTALL_RESULT=$?

        if [[ $INSTALL_RESULT -eq 0 ]]; then
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            if ! command -v uv &>/dev/null; then
                echo "uv installed but not in PATH. Please restart your terminal."
                exit 1
            fi
        else
            echo "Installation failed. Please install uv manually:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  or: brew install uv"
            exit 1
        fi
    else
        echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

echo "[Environment] Using uv package manager..."
echo

# Check if virtual environment exists
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "[Setup] Virtual environment not found. Setting up environment..."
    echo "This will take a few minutes on first run."
    echo
    echo "Running: uv sync"
    echo

    cd "$SCRIPT_DIR" && uv sync

    echo
    echo "========================================"
    echo "Environment setup completed!"
    echo "========================================"
    echo
fi

echo "Starting ACE-Step API Server (MLX backend)..."
echo

# Build command with optional parameters
CMD=(uv run acestep-api --host "$HOST" --port "$PORT")
[[ -n "$API_KEY" ]] && CMD+=(--api-key "$API_KEY")
[[ -n "$DOWNLOAD_SOURCE" ]] && CMD+=(--download-source "$DOWNLOAD_SOURCE")
[[ -n "$LM_MODEL_PATH" ]] && CMD+=(--lm-model-path "$LM_MODEL_PATH")

cd "$SCRIPT_DIR" && exec "${CMD[@]}"
