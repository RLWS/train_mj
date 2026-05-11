#!/usr/bin/env bash
set -Eeuo pipefail

# P800/Linux one-command pipeline.
#
# Common examples:
#   ./start.sh
#   TASKS=turn ./start.sh
#   TASKS=response ./start.sh
#   TASKS=multitask ./start.sh
#   RUN_VALIDATE=0 ./start.sh
#   RUN_TRAIN=0 RUN_EVAL=0 ./start.sh
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

# RAW_XML_DIR is usually the mounted dataset path on the training machine.
# Generated data, reports, checkpoints, and logs stay under the repository by default.
RAW_XML_DIR="${RAW_XML_DIR:-/workspace/dataset}"
DATA_ROOT="${DATA_ROOT:-data}"
BRONZE_EVENTS_DIR="${BRONZE_EVENTS_DIR:-$DATA_ROOT/bronze_events}"
SAMPLES_DIR="${SAMPLES_DIR:-$DATA_ROOT/silver_samples}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_ROOT/manifests}"
REPORT_DIR="${REPORT_DIR:-outputs/reports}"
EVENT_VALIDATION_REPORT="${EVENT_VALIDATION_REPORT:-$REPORT_DIR/event_validation.json}"

WORKERS="${WORKERS:-$(default_workers)}"

# TASKS controls train/eval tasks. Valid values: turn, response, multitask.
# Multiple tasks are space-separated, for example TASKS="turn response".
TASKS="${TASKS:-turn response}"
TURN_TRAIN_CONFIG="${TURN_TRAIN_CONFIG:-configs/turn_p800.yaml}"
RESPONSE_TRAIN_CONFIG="${RESPONSE_TRAIN_CONFIG:-configs/response_p800.yaml}"
MULTITASK_TRAIN_CONFIG="${MULTITASK_TRAIN_CONFIG:-configs/multitask_p800.yaml}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_BUILD_DATASET="${RUN_BUILD_DATASET:-1}"
RUN_MANIFEST="${RUN_MANIFEST:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_BUILD_VALIDATION="${SKIP_BUILD_VALIDATION:-0}"

EVAL_SPLIT="${EVAL_SPLIT:-valid}"
TURN_EVAL_CKPT="${TURN_EVAL_CKPT:-outputs/checkpoints/turn_p800/best.pt}"
RESPONSE_EVAL_CKPT="${RESPONSE_EVAL_CKPT:-outputs/checkpoints/response_p800/best.pt}"
MULTITASK_EVAL_CKPT="${MULTITASK_EVAL_CKPT:-outputs/checkpoints/multitask_p800_v1_1/best.pt}"
TURN_EVAL_REPORT="${TURN_EVAL_REPORT:-$REPORT_DIR/turn_p800_${EVAL_SPLIT}_eval.json}"
RESPONSE_EVAL_REPORT="${RESPONSE_EVAL_REPORT:-$REPORT_DIR/response_p800_${EVAL_SPLIT}_eval.json}"
MULTITASK_TURN_EVAL_REPORT="${MULTITASK_TURN_EVAL_REPORT:-$REPORT_DIR/multitask_p800_v1_1_turn_${EVAL_SPLIT}_eval.json}"
MULTITASK_RESPONSE_EVAL_REPORT="${MULTITASK_RESPONSE_EVAL_REPORT:-$REPORT_DIR/multitask_p800_v1_1_response_${EVAL_SPLIT}_eval.json}"

train_config_for_task() {
  case "$1" in
    turn)
      printf '%s\n' "$TURN_TRAIN_CONFIG"
      ;;
    response)
      printf '%s\n' "$RESPONSE_TRAIN_CONFIG"
      ;;
    multitask)
      printf '%s\n' "$MULTITASK_TRAIN_CONFIG"
      ;;
    *)
      echo "unsupported task in TASKS: $1 (expected turn, response, or multitask)" >&2
      exit 1
      ;;
  esac
}

eval_ckpt_for_task() {
  case "$1" in
    turn)
      printf '%s\n' "$TURN_EVAL_CKPT"
      ;;
    response)
      printf '%s\n' "$RESPONSE_EVAL_CKPT"
      ;;
    multitask)
      printf '%s\n' "$MULTITASK_EVAL_CKPT"
      ;;
    *)
      echo "unsupported task in TASKS: $1 (expected turn, response, or multitask)" >&2
      exit 1
      ;;
  esac
}

eval_report_for_task() {
  case "$1" in
    turn)
      printf '%s\n' "$TURN_EVAL_REPORT"
      ;;
    response)
      printf '%s\n' "$RESPONSE_EVAL_REPORT"
      ;;
    multitask)
      printf '%s\n' "$MULTITASK_TURN_EVAL_REPORT"
      ;;
    *)
      echo "unsupported task in TASKS: $1 (expected turn, response, or multitask)" >&2
      exit 1
      ;;
  esac
}

mkdir -p "$BRONZE_EVENTS_DIR" "$SAMPLES_DIR" "$MANIFEST_DIR" "$REPORT_DIR" outputs/checkpoints outputs/logs

log "pipeline settings"
printf '  RAW_XML_DIR=%s\n' "$RAW_XML_DIR"
printf '  BRONZE_EVENTS_DIR=%s\n' "$BRONZE_EVENTS_DIR"
printf '  SAMPLES_DIR=%s\n' "$SAMPLES_DIR"
printf '  MANIFEST_DIR=%s\n' "$MANIFEST_DIR"
printf '  WORKERS=%s\n' "$WORKERS"
printf '  TASKS=%s\n' "$TASKS"
printf '  RUN_VALIDATE=%s\n' "$RUN_VALIDATE"
printf '  RUN_TRAIN=%s\n' "$RUN_TRAIN"
printf '  RUN_EVAL=%s\n' "$RUN_EVAL"

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
  # If validate-events already ran, do not replay-validate every shard again.
  # If validation was skipped but build-time validation is desired, keep SKIP_BUILD_VALIDATION=0.
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
  for task in $TASKS; do
    config="$(train_config_for_task "$task")"
    if [ ! -f "$config" ]; then
      echo "missing $task train config: $config" >&2
      exit 1
    fi
    run "$PYTHON_BIN" -m mjai.cli train --config "$config"
  done
else
  log "skip train"
fi

if [ "$RUN_EVAL" = "1" ]; then
  for task in $TASKS; do
    ckpt="$(eval_ckpt_for_task "$task")"
    if [ ! -f "$ckpt" ]; then
      echo "missing $task eval checkpoint: $ckpt" >&2
      exit 1
    fi
    if [ "$task" = "multitask" ]; then
      mkdir -p "$(dirname "$MULTITASK_TURN_EVAL_REPORT")" "$(dirname "$MULTITASK_RESPONSE_EVAL_REPORT")"
      run "$PYTHON_BIN" -m mjai.cli eval \
        --ckpt "$ckpt" \
        --task turn \
        --split "$EVAL_SPLIT" \
        --report "$MULTITASK_TURN_EVAL_REPORT"
      run "$PYTHON_BIN" -m mjai.cli eval \
        --ckpt "$ckpt" \
        --task response \
        --split "$EVAL_SPLIT" \
        --report "$MULTITASK_RESPONSE_EVAL_REPORT"
    else
      report="$(eval_report_for_task "$task")"
      mkdir -p "$(dirname "$report")"
      run "$PYTHON_BIN" -m mjai.cli eval \
        --ckpt "$ckpt" \
        --split "$EVAL_SPLIT" \
        --report "$report"
    fi
  done
else
  log "skip eval"
fi

log "pipeline finished"
