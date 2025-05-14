#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"/..

SCHEMA_DIR=schema
BASE_TARGET_DIR=src/model_explorer_tosa_flatbuffer

for schema in "$SCHEMA_DIR"/*.fbs; do
  schema_basename=$(basename "$schema" .fbs)
  target_dir="$BASE_TARGET_DIR/$schema_basename"

  mkdir -p "$target_dir"
  flatc --python --gen-onefile --gen-object-api --python-typing \
    -o "$target_dir" "$schema"
done

