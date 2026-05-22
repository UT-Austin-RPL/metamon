#!/usr/bin/env bash
# Flatten a self-play trajectory pile from the ladder structure:
#   {pile_dir}/{model_name}/metamon/{format}/*.json.lz4
# into the MetamonDataset-expected structure:
#   {pile_dir}/{format}/*.json.lz4
#
# Then generates a fresh index.csv.
#
# Usage:
#   bash tools/flatten_self_play_pile.sh /path/to/pile                # dry run (default)
#   bash tools/flatten_self_play_pile.sh /path/to/pile --execute      # actually do it
set -euo pipefail

PILE_DIR="${1:?Usage: $0 <pile_dir> [--execute]}"
EXECUTE=false
if [[ "${2:-}" == "--execute" ]]; then
    EXECUTE=true
fi

PILE_DIR="${PILE_DIR%/}"

if [[ ! -d "$PILE_DIR" ]]; then
    echo "ERROR: Directory not found: $PILE_DIR"
    exit 1
fi

# Discover model subdirs (dirs that contain a metamon/ subfolder)
MODEL_DIRS=()
for d in "$PILE_DIR"/*/; do
    if [[ -d "${d}metamon" ]]; then
        MODEL_DIRS+=("$d")
    fi
done

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "No model subdirs with metamon/ found in $PILE_DIR"
    echo "Directory already flat — regenerating index.csv..."
    rm -f "$PILE_DIR/index.csv"
    INDEX_FILE="$PILE_DIR/index.csv"
    echo "filename" > "$INDEX_FILE"
    for format_dir in "$PILE_DIR"/*/; do
        [[ -d "$format_dir" ]] || continue
        format_name="$(basename "$format_dir")"
        find "$format_dir" -maxdepth 1 \( -name '*.json.lz4' -o -name '*.json' \) -printf "${format_name}/%f\n" \
            >> "$INDEX_FILE"
    done
    INDEX_COUNT=$(( $(wc -l < "$INDEX_FILE") - 1 ))
    echo "  Wrote $INDEX_COUNT entries to $INDEX_FILE"
    exit 0
fi

echo "============================================================"
echo "  Flatten Self-Play Pile"
echo "============================================================"
echo "  Pile dir:    $PILE_DIR"
echo "  Mode:        $(if $EXECUTE; then echo 'EXECUTE'; else echo 'DRY RUN'; fi)"
echo "  Model dirs:  ${#MODEL_DIRS[@]}"
echo ""

# Discover all format dirs and count files
declare -A FORMAT_COUNTS
TOTAL_FILES=0

for model_dir in "${MODEL_DIRS[@]}"; do
    model_name="$(basename "$model_dir")"
    for format_dir in "$model_dir"metamon/*/; do
        [[ -d "$format_dir" ]] || continue
        format_name="$(basename "$format_dir")"
        count=$(find "$format_dir" -maxdepth 1 -name '*.json.lz4' -o -name '*.json' | wc -l)
        FORMAT_COUNTS["$format_name"]=$(( ${FORMAT_COUNTS["$format_name"]:-0} + count ))
        TOTAL_FILES=$(( TOTAL_FILES + count ))
        echo "  $model_name / $format_name : $count files"
    done
done

echo ""
echo "  Total files to move: $TOTAL_FILES"
echo ""

# Show where files will land
echo "  Destination directories:"
for format_name in "${!FORMAT_COUNTS[@]}"; do
    dest="$PILE_DIR/$format_name"
    count=${FORMAT_COUNTS[$format_name]}
    if [[ -d "$dest" ]]; then
        existing=$(find "$dest" -maxdepth 1 -name '*.json.lz4' -o -name '*.json' | wc -l)
        echo "    $dest/ : $count new files (+ $existing existing)"
    else
        echo "    $dest/ : $count files (will be created)"
    fi
done

echo ""

# Check for filename collisions
echo "  Checking for filename collisions..."
COLLISION_COUNT=0
for model_dir in "${MODEL_DIRS[@]}"; do
    for format_dir in "$model_dir"metamon/*/; do
        [[ -d "$format_dir" ]] || continue
        format_name="$(basename "$format_dir")"
        dest="$PILE_DIR/$format_name"
        if [[ -d "$dest" ]]; then
            while IFS= read -r filepath; do
                fname="$(basename "$filepath")"
                if [[ -e "$dest/$fname" ]]; then
                    COLLISION_COUNT=$((COLLISION_COUNT + 1))
                    if [[ $COLLISION_COUNT -le 5 ]]; then
                        echo "    COLLISION: $fname already exists in $dest/"
                    fi
                fi
            done < <(find "$format_dir" -maxdepth 1 \( -name '*.json.lz4' -o -name '*.json' \))
        fi
    done
done

if [[ $COLLISION_COUNT -gt 0 ]]; then
    echo "  WARNING: $COLLISION_COUNT filename collisions detected!"
    echo "  Colliding files will be SKIPPED (not overwritten)."
else
    echo "  No collisions found."
fi

echo ""

if ! $EXECUTE; then
    echo "  DRY RUN complete. Re-run with --execute to apply."
    exit 0
fi

echo "  Moving files..."

for model_dir in "${MODEL_DIRS[@]}"; do
    model_name="$(basename "$model_dir")"
    for format_dir in "$model_dir"metamon/*/; do
        [[ -d "$format_dir" ]] || continue
        format_name="$(basename "$format_dir")"
        dest="$PILE_DIR/$format_name"
        mkdir -p "$dest"

        if [[ $COLLISION_COUNT -eq 0 ]]; then
            find "$format_dir" -maxdepth 1 \( -name '*.json.lz4' -o -name '*.json' \) \
                -exec mv -t "$dest" {} +
        else
            find "$format_dir" -maxdepth 1 \( -name '*.json.lz4' -o -name '*.json' \) \
                -exec mv -n -t "$dest" {} +
        fi
    done
    echo "    $model_name done"
done

# Remove empty model dirs
echo "  Cleaning up empty model directories..."
for model_dir in "${MODEL_DIRS[@]}"; do
    find "$model_dir" -type d -empty -delete 2>/dev/null || true
    if [[ -d "$model_dir" ]]; then
        rmdir --ignore-fail-on-non-empty -p "$model_dir"metamon 2>/dev/null || true
    fi
    if [[ ! -d "$model_dir" ]]; then
        echo "    Removed $(basename "$model_dir")/"
    else
        echo "    $(basename "$model_dir")/ still has files, kept"
    fi
done

# Remove stale index.csv so MetamonDataset rebuilds it on next load
if [[ -f "$PILE_DIR/index.csv" ]]; then
    rm "$PILE_DIR/index.csv"
    echo "  Removed stale index.csv"
fi

# Generate fresh index.csv
echo ""
echo "  Generating fresh index.csv..."
INDEX_FILE="$PILE_DIR/index.csv"
echo "filename" > "$INDEX_FILE"
for format_name in "${!FORMAT_COUNTS[@]}"; do
    format_dir="$PILE_DIR/$format_name"
    [[ -d "$format_dir" ]] || continue
    find "$format_dir" -maxdepth 1 \( -name '*.json.lz4' -o -name '*.json' \) -printf "${format_name}/%f\n" \
        >> "$INDEX_FILE"
done
INDEX_COUNT=$(( $(wc -l < "$INDEX_FILE") - 1 ))
echo "  Wrote $INDEX_COUNT entries to $INDEX_FILE"

echo ""
echo "  Done!"
