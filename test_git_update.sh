#!/usr/bin/env bash
# Test Git Update Check Functionality - Linux/macOS
# This script tests the update check without actually starting the application

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Test Git Update Check"
echo "========================================"
echo

# Test 1: Check if git is available
echo "[Test 1] Checking git..."
if command -v git &>/dev/null; then
    echo "[PASS] git found"
    git --version
else
    echo "[FAIL] git not found"
    echo
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Please install git:"
        echo "  xcode-select --install"
        echo "  or: brew install git"
    else
        echo "Please install git:"
        echo "  Ubuntu/Debian: sudo apt install git"
        echo "  CentOS/RHEL:   sudo yum install git"
        echo "  Arch:          sudo pacman -S git"
    fi
    echo
    echo "========================================"
    echo "Test Failed"
    echo "========================================"
    exit 1
fi
echo

# Test 2: Check if this is a git repository
echo "[Test 2] Checking git repository..."
cd "$SCRIPT_DIR"
if git rev-parse --git-dir &>/dev/null; then
    echo "[PASS] Valid git repository"
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
    CURRENT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    echo "  Branch: $CURRENT_BRANCH"
    echo "  Commit: $CURRENT_COMMIT"
else
    echo "[FAIL] Not a git repository"
    echo
    echo "This directory is not a git repository."
    echo "Please clone from GitHub:"
    echo "  git clone https://github.com/ace-step/ACE-Step-1.5.git"
    echo
    echo "========================================"
    echo "Test Failed"
    echo "========================================"
    exit 1
fi
echo

# Test 3: Check if check_update.sh exists
echo "[Test 3] Checking check_update.sh..."
if [[ -f "$SCRIPT_DIR/check_update.sh" ]]; then
    echo "[PASS] check_update.sh found"
else
    echo "[FAIL] check_update.sh not found"
    echo
    echo "========================================"
    echo "Test Failed"
    echo "========================================"
    exit 1
fi
echo

# Test 4: Run update check
echo "[Test 4] Running update check..."
echo "========================================"
echo

UPDATE_RESULT=0
bash "$SCRIPT_DIR/check_update.sh" || UPDATE_RESULT=$?

echo
echo "========================================"
echo

case $UPDATE_RESULT in
    0)
        echo "[Test 4] Update check completed successfully"
        echo "  Result: Already up to date or updated"
        ;;
    1)
        echo "[Test 4] Update check failed"
        echo "  Result: Error occurred"
        ;;
    2)
        echo "[Test 4] Update check skipped"
        echo "  Result: Network timeout"
        ;;
    *)
        echo "[Test 4] Unknown result: $UPDATE_RESULT"
        ;;
esac
echo

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo

if [[ $UPDATE_RESULT -le 2 ]]; then
    echo "[PASS] All tests completed"
    echo
    echo "The update check feature is working correctly."
    echo "You can now enable it in the launcher scripts:"
    echo "  CHECK_UPDATE=\"true\""
else
    echo "[FAIL] Some tests failed"
    echo
    echo "========================================"
    echo "Test Failed"
    echo "========================================"
    exit 1
fi
