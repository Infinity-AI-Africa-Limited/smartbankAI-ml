#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/opt/smartbank-src
MOUNT_ROOT=/workspace

cd "$SOURCE_ROOT"
bash scripts/build_synthetic_models.sh

mkdir -p "$MOUNT_ROOT/data"
rm -rf "$MOUNT_ROOT/data/synthetic"
cp -a "$SOURCE_ROOT/data/synthetic" "$MOUNT_ROOT/data/synthetic"

for source_models in "$SOURCE_ROOT"/agents/*/models; do
  agent_dir="$(dirname "$source_models")"
  agent_name="$(basename "$agent_dir")"
  target_models="$MOUNT_ROOT/agents/$agent_name/models"
  mkdir -p "$target_models"
  rm -rf "$target_models"/*
  cp -a "$source_models"/. "$target_models"/
done

echo "Synthetic datasets and model artefacts copied to $MOUNT_ROOT"
