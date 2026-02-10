#!/usr/bin/env bash
# Config Merge Helper - Linux/macOS
# This script helps merge backed up files with new version

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "ACE-Step Backup Merge Helper"
echo "========================================"
echo

# Check for backup directories
FOUND_BACKUPS=0
BACKUP_DIRS=()

echo "Searching for backup directories..."
echo

for dir in "$SCRIPT_DIR"/.update_backup_*; do
    [[ -d "$dir" ]] || continue
    FOUND_BACKUPS=1
    BACKUP_DIRS+=("$dir")
    echo "Found backup: $(basename "$dir")"
    echo "  Location: $dir"
    echo "  Files:"
    find "$dir" -type f | while read -r f; do
        echo "    - ${f#$dir/}"
    done
    echo
done

if [[ $FOUND_BACKUPS -eq 0 ]]; then
    echo "No backup directories found."
    echo
    echo "Backups are created when updates conflict with your local changes."
    exit 0
fi

echo "========================================"
echo "Merge Options"
echo "========================================"
echo
echo "1. Compare backup with current files"
echo "2. Restore a file from backup (overwrites current)"
echo "3. List all backed up files"
echo "4. Delete old backups"
echo "5. Exit"
echo

read -rp "Select option (1-5): " CHOICE

case "$CHOICE" in
    1)
        echo
        echo "========================================"
        echo "Compare Files"
        echo "========================================"
        echo
        echo "Available backup directories:"
        for i in "${!BACKUP_DIRS[@]}"; do
            echo "  $((i + 1)). $(basename "${BACKUP_DIRS[$i]}")"
        done
        echo

        read -rp "Select backup number: " BACKUP_CHOICE
        BACKUP_IDX=$((BACKUP_CHOICE - 1))

        if [[ $BACKUP_IDX -lt 0 || $BACKUP_IDX -ge ${#BACKUP_DIRS[@]} ]]; then
            echo "Invalid selection."
            exit 1
        fi

        SELECTED_BACKUP="${BACKUP_DIRS[$BACKUP_IDX]}"
        echo
        echo "Files in backup:"
        find "$SELECTED_BACKUP" -type f | while read -r f; do
            echo "  - ${f#$SELECTED_BACKUP/}"
        done
        echo

        read -rp "Enter filename to compare (e.g., start_gradio_ui.sh or acestep/handler.py): " FILE_NAME

        BACKUP_FILE="$SELECTED_BACKUP/$FILE_NAME"
        CURRENT_FILE="$SCRIPT_DIR/$FILE_NAME"

        if [[ ! -f "$BACKUP_FILE" ]]; then
            echo "Backup file not found: $BACKUP_FILE"
            exit 1
        fi

        if [[ ! -f "$CURRENT_FILE" ]]; then
            echo "Current file not found: $CURRENT_FILE"
            exit 1
        fi

        echo
        echo "Comparing files..."
        echo "Backup version:  $BACKUP_FILE"
        echo "Current version: $CURRENT_FILE"
        echo

        # Use diff to compare
        if command -v colordiff &>/dev/null; then
            colordiff -u "$BACKUP_FILE" "$CURRENT_FILE" || true
        else
            diff -u "$BACKUP_FILE" "$CURRENT_FILE" || true
        fi
        ;;

    2)
        echo
        echo "========================================"
        echo "Restore File from Backup"
        echo "========================================"
        echo
        echo "[Warning] This will OVERWRITE the current file!"
        echo
        echo "Available backup directories:"
        for i in "${!BACKUP_DIRS[@]}"; do
            echo "  $((i + 1)). $(basename "${BACKUP_DIRS[$i]}")"
        done
        echo

        read -rp "Select backup number: " BACKUP_CHOICE
        BACKUP_IDX=$((BACKUP_CHOICE - 1))

        if [[ $BACKUP_IDX -lt 0 || $BACKUP_IDX -ge ${#BACKUP_DIRS[@]} ]]; then
            echo "Invalid selection."
            exit 1
        fi

        SELECTED_BACKUP="${BACKUP_DIRS[$BACKUP_IDX]}"
        echo
        echo "Files in backup:"
        find "$SELECTED_BACKUP" -type f | while read -r f; do
            echo "  - ${f#$SELECTED_BACKUP/}"
        done
        echo

        read -rp "Enter filename to restore (e.g., start_gradio_ui.sh or acestep/handler.py): " FILE_NAME

        BACKUP_FILE="$SELECTED_BACKUP/$FILE_NAME"
        CURRENT_FILE="$SCRIPT_DIR/$FILE_NAME"

        if [[ ! -f "$BACKUP_FILE" ]]; then
            echo "Backup file not found: $BACKUP_FILE"
            exit 1
        fi

        echo
        echo "About to restore:"
        echo "  From: $BACKUP_FILE"
        echo "  To:   $CURRENT_FILE"
        echo

        read -rp "Are you sure? This will overwrite the current file. (Y/N): " CONFIRM
        if [[ "${CONFIRM^^}" == "Y" ]]; then
            cp "$BACKUP_FILE" "$CURRENT_FILE"
            echo
            echo "[Success] File restored successfully."
        else
            echo
            echo "Restore cancelled."
        fi
        ;;

    3)
        echo
        echo "========================================"
        echo "All Backed Up Files"
        echo "========================================"
        echo

        for dir in "${BACKUP_DIRS[@]}"; do
            echo "Backup: $(basename "$dir")"
            echo "Location: $dir"
            echo "Files:"
            find "$dir" -type f | while read -r f; do
                echo "  - ${f#$dir/}"
            done
            echo
        done
        ;;

    4)
        echo
        echo "========================================"
        echo "Delete Old Backups"
        echo "========================================"
        echo
        echo "[Warning] This will permanently delete backup directories!"
        echo
        echo "Available backups:"
        for dir in "${BACKUP_DIRS[@]}"; do
            echo "  - $(basename "$dir")"
        done
        echo

        read -rp "Delete all backups? (Y/N): " DELETE_CONFIRM
        if [[ "${DELETE_CONFIRM^^}" == "Y" ]]; then
            echo
            echo "Deleting backups..."
            for dir in "${BACKUP_DIRS[@]}"; do
                echo "  Deleting: $(basename "$dir")"
                rm -rf "$dir"
            done
            echo
            echo "[Done] Backups deleted."
        else
            echo
            echo "Deletion cancelled."
        fi
        ;;

    5)
        exit 0
        ;;

    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
