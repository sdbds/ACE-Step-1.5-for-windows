#!/usr/bin/env bash
# Install uv Package Manager
# This script installs uv using the official installer
#
# Usage:
#   ./install_uv.sh           - Interactive mode (default)
#   ./install_uv.sh --silent  - Silent mode for script calls
#
# Exit codes:
#   0 - Success (uv installed and available)
#   1 - Installation failed
#   2 - User cancelled (interactive mode only)

set -euo pipefail

SILENT_MODE=0
if [[ "${1:-}" == "--silent" || "${1:-}" == "-s" ]]; then
    SILENT_MODE=1
fi

log() {
    if [[ "$SILENT_MODE" -eq 0 ]]; then
        echo "$@"
    fi
}

# Check if uv is already installed
if command -v uv &>/dev/null; then
    if [[ "$SILENT_MODE" -eq 1 ]]; then
        exit 0
    fi

    echo "uv is already installed!"
    echo "Current version:"
    uv --version
    echo
    echo "Installation location:"
    command -v uv
    echo

    read -rp "Reinstall uv? (Y/N): " REINSTALL
    if [[ "${REINSTALL^^}" != "Y" ]]; then
        echo
        echo "Installation cancelled."
        exit 2
    fi
    echo
fi

log "Installing uv..."
log

# Try the official installer (works on both Linux and macOS)
log "Using official installer (curl)..."

if command -v curl &>/dev/null; then
    if [[ "$SILENT_MODE" -eq 1 ]]; then
        curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
    else
        log "Downloading uv installer..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
    fi
elif command -v wget &>/dev/null; then
    log "curl not found, trying wget..."
    if [[ "$SILENT_MODE" -eq 1 ]]; then
        wget -qO- https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
    else
        wget -qO- https://astral.sh/uv/install.sh | sh || true
    fi
else
    log "========================================"
    log "ERROR: Neither curl nor wget found!"
    log "========================================"
    log
    log "Please install curl or wget first:"
    log
    if [[ "$(uname)" == "Darwin" ]]; then
        log "  brew install curl"
    else
        log "  Ubuntu/Debian: sudo apt install curl"
        log "  CentOS/RHEL:   sudo yum install curl"
        log "  Arch:          sudo pacman -S curl"
    fi
    log
    exit 1
fi

# Update PATH to include common uv install locations
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Verify installation
if command -v uv &>/dev/null; then
    log
    log "========================================"
    log "Installation successful!"
    log "========================================"
    log
    log "uv version:"
    log "$(uv --version)"
    log
    log "Installation location:"
    log "$(command -v uv)"
    log
    log "You can now use ACE-Step by running:"
    log "  ./start_gradio_ui.sh"
    log "  ./start_api_server.sh"
    if [[ "$(uname)" == "Darwin" ]]; then
        log "  ./start_gradio_ui_macos.sh  (Apple Silicon with MLX)"
        log "  ./start_api_server_macos.sh (Apple Silicon with MLX)"
    fi
    log
    exit 0
fi

# Check default installation location
if [[ -x "$HOME/.local/bin/uv" ]]; then
    log
    log "========================================"
    log "Installation successful!"
    log "========================================"
    log
    log "Installation location: $HOME/.local/bin/uv"
    log
    log "NOTE: uv is not in your PATH yet."
    log "Add to your shell profile:"
    log "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    log "  source ~/.bashrc"
    log
    exit 0
fi

if [[ -x "$HOME/.cargo/bin/uv" ]]; then
    log
    log "========================================"
    log "Installation successful!"
    log "========================================"
    log
    log "Installation location: $HOME/.cargo/bin/uv"
    log
    log "NOTE: uv is not in your PATH yet."
    log "Add to your shell profile:"
    log "  echo 'export PATH=\"\$HOME/.cargo/bin:\$PATH\"' >> ~/.bashrc"
    log "  source ~/.bashrc"
    log
    exit 0
fi

# Installation failed
log
log "========================================"
log "ERROR: Installation failed!"
log "========================================"
log
log "Please install uv manually:"
log
log "  curl -LsSf https://astral.sh/uv/install.sh | sh"
log
if [[ "$(uname)" == "Darwin" ]]; then
    log "Or using Homebrew:"
    log "  brew install uv"
fi
log
exit 1
