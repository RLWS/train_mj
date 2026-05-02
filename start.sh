#!/usr/bin/env bash
set -Eeuo pipefail

# P800/Linux default:
#   RAW_XML_DIR=/workspace/dataset WORKERS=32 ./start.sh
# Resume from prepared data:
#   RUN_PREPROCESS=0 RUN_VALIDATE=0 SKIP_BUILD_VALIDATION=1 ./start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

default_workers() {
  local workers=1
  if command -v nproc >/dev/null 2>&1; then
    workers="$(nproc)"
  fi
  if [ "$workers" -gt 32 ]; then
    workers=32
  fi
  echo "$workers"
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  log "$*"
  "$@"
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    echo "missing $label directory: $path" >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_XML_DIR="${RAW_XML_DIR:-/workspace/dataset}"
DATA_ROOT="${DATA_ROOT:-data}"
BRONZE_EVENTS_DIR="${BRONZE_EVENTS_DIR:-$DATA_ROOT/bronze_events}"
SAMPLES_DIR="${SAMPLES_DIR:-$DATA_ROOT/silver_samples}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_ROOT/manifests}"
REPORT_DIR="${REPORT_DIR:-outputs/reports}"
EVENT_VALIDATION_REPORT="${EVENT_VALIDATION_REPORT:-$REPORT_DIR/event_validation.json}"

WORKERS="${WORKERS:-$(default_workers)}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-configs/turn_p800.yaml configs/response_p800.yaml}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_BUILD_DATASET="${RUN_BUILD_DATASET:-1}"
RUN_MANIFEST="${RUN_MANIFEST:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
SKIP_BUILD_VALIDATION="${SKIP_BUILD_VALIDATION:-0}"

EVAL_SPLIT="${EVAL_SPLIT:-valid}"
EVAL_CKPTS="${EVAL_CKPTS:-outputs/checkpoints/turn_p800/best.pt outputs/checkpoints/response_p800/best.pt}"

mkdir -p "$BRONZE_EVENTS_DIR" "$SAMPLES_DIR" "$MANIFEST_DIR" "$REPORT_DIR" outputs/checkpoints outputs/logs

log "pipeline settings"
printf '  RAW_XML_DIR=%s\n' "$RAW_XML_DIR"
printf '  BRONZE_EVENTS_DIR=%s\n' "$BRONZE_EVENTS_DIR"
printf '  SAMPLES_DIR=%s\n' "$SAMPLES_DIR"
printf '  MANIFEST_DIR=%s\n' "$MANIFEST_DIR"
printf '  WORKERS=%s\n' "$WORKERS"
printf '  TRAIN_CONFIGS=%s\n' "$TRAIN_CONFIGS"

if [ "$RUN_PREPROCESS" = "1" ]; then
  require_dir "$RAW_XML_DIR" "raw XML"
  run "$PYTHON_BIN" -m mjai.cli preprocess \
    --input "$RAW_XML_DIR" \
    --output "$BRONZE_EVENTS_DIR"
else
  log "skip preprocess"
fi

if [ "$RUN_VALIDATE" = "1" ]; then
  require_dir "$BRONZE_EVENTS_DIR" "bronze events"
  run "$PYTHON_BIN" -m mjai.cli validate-events \
    --input "$BRONZE_EVENTS_DIR" \
    --report "$EVENT_VALIDATION_REPORT"
else
  log "skip validate-events"
fi

if [ "$RUN_BUILD_DATASET" = "1" ]; then
  require_dir "$BRONZE_EVENTS_DIR" "bronze events"
  build_args=(
    -m mjai.cli build-dataset
    --input "$BRONZE_EVENTS_DIR"
    --output "$SAMPLES_DIR"
    --workers "$WORKERS"
  )
  if [ "$RUN_VALIDATE" = "1" ] || [ "$SKIP_BUILD_VALIDATION" = "1" ]; then
    build_args+=(--skip-validation)
  fi
  run "$PYTHON_BIN" "${build_args[@]}"
else
  log "skip build-dataset"
fi

if [ "$RUN_MANIFEST" = "1" ]; then
  require_dir "$SAMPLES_DIR" "silver samples"
  run "$PYTHON_BIN" -m mjai.cli make-manifest \
    --input "$SAMPLES_DIR" \
    --output "$MANIFEST_DIR"
else
  log "skip make-manifest"
fi

if [ "$RUN_TRAIN" = "1" ]; then
  for config in $TRAIN_CONFIGS; do
    if [ ! -f "$config" ]; then
      echo "missing train config: $config" >&2
      exit 1
    fi
    run "$PYTHON_BIN" -m mjai.cli train --config "$config"
  done
else
  log "skip train"
fi

if [ "$RUN_EVAL" = "1" ]; then
  for ckpt in $EVAL_CKPTS; do
    if [ ! -f "$ckpt" ]; then
      echo "missing eval checkpoint: $ckpt" >&2
      exit 1
    fi
    name="$(basename "$(dirname "$ckpt")")"
    run "$PYTHON_BIN" -m mjai.cli eval \
      --ckpt "$ckpt" \
      --split "$EVAL_SPLIT" \
      --report "$REPORT_DIR/${name}_${EVAL_SPLIT}_eval.json"
  done
else
  log "skip eval"
fi

log "pipeline finished"
