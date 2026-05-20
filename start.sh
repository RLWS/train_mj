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
#   TASKS=multitask RUN_EXTRA_EVAL=1 RUN_CALIBRATION=1 EVAL_SPLIT=test ./start.sh
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

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "missing $label file: $path" >&2
    exit 1
  fi
}

contains_task() {
  local wanted="$1"
  local task
  for task in $TASKS; do
    if [ "$task" = "$wanted" ]; then
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-python}"
DDP_NPROC_PER_NODE="${DDP_NPROC_PER_NODE:-}"

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
RUN_EXTRA_EVAL="${RUN_EXTRA_EVAL:-0}"
RUN_CALIBRATION="${RUN_CALIBRATION:-0}"
SKIP_BUILD_VALIDATION="${SKIP_BUILD_VALIDATION:-0}"

EVAL_SPLIT="${EVAL_SPLIT:-valid}"
TURN_EVAL_CKPT="${TURN_EVAL_CKPT:-outputs/checkpoints/turn_p800/best.pt}"
RESPONSE_EVAL_CKPT="${RESPONSE_EVAL_CKPT:-outputs/checkpoints/response_p800/best.pt}"
MULTITASK_EVAL_CKPT="${MULTITASK_EVAL_CKPT:-outputs/checkpoints/multitask_p800_v1_1/best.pt}"
RESPONSE_CHECKPOINT_DIR="${RESPONSE_CHECKPOINT_DIR:-$(dirname "$RESPONSE_EVAL_CKPT")}"
MULTITASK_CHECKPOINT_DIR="${MULTITASK_CHECKPOINT_DIR:-$(dirname "$MULTITASK_EVAL_CKPT")}"
RESPONSE_REPORT_PREFIX="${RESPONSE_REPORT_PREFIX:-$(basename "$RESPONSE_CHECKPOINT_DIR")}"
MULTITASK_REPORT_PREFIX="${MULTITASK_REPORT_PREFIX:-$(basename "$MULTITASK_CHECKPOINT_DIR")}"
RESPONSE_BEST_CKPT="${RESPONSE_BEST_CKPT:-$RESPONSE_CHECKPOINT_DIR/best.pt}"
RESPONSE_BEST_RESPONSE_CKPT="${RESPONSE_BEST_RESPONSE_CKPT:-$RESPONSE_CHECKPOINT_DIR/best_response.pt}"
MULTITASK_BEST_CKPT="${MULTITASK_BEST_CKPT:-$MULTITASK_CHECKPOINT_DIR/best.pt}"
MULTITASK_BEST_TURN_CKPT="${MULTITASK_BEST_TURN_CKPT:-$MULTITASK_CHECKPOINT_DIR/best_turn.pt}"
MULTITASK_BEST_RESPONSE_CKPT="${MULTITASK_BEST_RESPONSE_CKPT:-$MULTITASK_CHECKPOINT_DIR/best_response.pt}"
TURN_EVAL_REPORT="${TURN_EVAL_REPORT:-$REPORT_DIR/turn_p800_${EVAL_SPLIT}_eval.json}"
RESPONSE_EVAL_REPORT="${RESPONSE_EVAL_REPORT:-$REPORT_DIR/response_p800_${EVAL_SPLIT}_eval.json}"
MULTITASK_TURN_EVAL_REPORT="${MULTITASK_TURN_EVAL_REPORT:-$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_turn_${EVAL_SPLIT}_eval.json}"
MULTITASK_RESPONSE_EVAL_REPORT="${MULTITASK_RESPONSE_EVAL_REPORT:-$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_response_${EVAL_SPLIT}_eval.json}"
EXTRA_EVAL_SPLITS="${EXTRA_EVAL_SPLITS:-$EVAL_SPLIT}"
CALIBRATION_SPLITS="${CALIBRATION_SPLITS:-valid test}"
CALIBRATION_PASS_BIASES="${CALIBRATION_PASS_BIASES:-0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40,0.60,0.80,1.00}"
CALIBRATION_FALSE_CALL_TARGETS="${CALIBRATION_FALSE_CALL_TARGETS:-1%,0.8%,0.5%}"
CALIBRATION_LOG_EVERY_N_BATCHES="${CALIBRATION_LOG_EVERY_N_BATCHES:-100}"

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

run_eval_report() {
  local ckpt="$1"
  local split="$2"
  local report="$3"
  local task_arg="${4:-}"
  require_file "$ckpt" "eval checkpoint"
  mkdir -p "$(dirname "$report")"
  if [ -n "$task_arg" ]; then
    run "$PYTHON_BIN" -m mjai.cli eval \
      --ckpt "$ckpt" \
      --task "$task_arg" \
      --split "$split" \
      --report "$report"
  else
    run "$PYTHON_BIN" -m mjai.cli eval \
      --ckpt "$ckpt" \
      --split "$split" \
      --report "$report"
  fi
}

run_response_calibration() {
  local ckpt="$1"
  local split="$2"
  local report="$3"
  local manifest="$MANIFEST_DIR/$split.jsonl"
  require_file "$ckpt" "calibration checkpoint"
  require_file "$manifest" "$split manifest"
  mkdir -p "$(dirname "$report")"
  run "$PYTHON_BIN" -m mjai.cli calibrate-response \
    --ckpt "$ckpt" \
    --split "$split" \
    --manifest "$manifest" \
    --report "$report" \
    --pass-biases "$CALIBRATION_PASS_BIASES" \
    --false-call-targets "$CALIBRATION_FALSE_CALL_TARGETS" \
    --log-every-n-batches "$CALIBRATION_LOG_EVERY_N_BATCHES"
}

config_requests_ddp() {
  local config="$1"
  local value
  value="$("$PYTHON_BIN" -c 'from mjai.utils.config import load_resolved_yaml; import sys; print(str(bool(load_resolved_yaml(sys.argv[1]).get("train", {}).get("distributed", False))).lower())' "$config")"
  [ "$value" = "true" ]
}

default_ddp_nproc() {
  if [ -n "$DDP_NPROC_PER_NODE" ]; then
    printf '%s\n' "$DDP_NPROC_PER_NODE"
    return
  fi
  "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)'
}

run_train_config() {
  local config="$1"
  if config_requests_ddp "$config"; then
    local nproc
    nproc="$(default_ddp_nproc)"
    if [ "$nproc" -le 1 ]; then
      echo "config requests DDP but DDP_NPROC_PER_NODE/default CUDA device count is not > 1: $config" >&2
      exit 1
    fi
    run "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      "--nproc_per_node=$nproc" \
      -m mjai.cli train \
      --config "$config"
  else
    run "$PYTHON_BIN" -m mjai.cli train --config "$config"
  fi
}

mkdir -p "$BRONZE_EVENTS_DIR" "$SAMPLES_DIR" "$MANIFEST_DIR" "$REPORT_DIR" outputs/checkpoints outputs/logs

log "pipeline settings"
printf '  RAW_XML_DIR=%s\n' "$RAW_XML_DIR"
printf '  BRONZE_EVENTS_DIR=%s\n' "$BRONZE_EVENTS_DIR"
printf '  SAMPLES_DIR=%s\n' "$SAMPLES_DIR"
printf '  MANIFEST_DIR=%s\n' "$MANIFEST_DIR"
printf '  WORKERS=%s\n' "$WORKERS"
printf '  TASKS=%s\n' "$TASKS"
printf '  DDP_NPROC_PER_NODE=%s\n' "${DDP_NPROC_PER_NODE:-auto}"
printf '  RUN_VALIDATE=%s\n' "$RUN_VALIDATE"
printf '  RUN_TRAIN=%s\n' "$RUN_TRAIN"
printf '  RUN_EVAL=%s\n' "$RUN_EVAL"
printf '  RUN_EXTRA_EVAL=%s\n' "$RUN_EXTRA_EVAL"
printf '  RUN_CALIBRATION=%s\n' "$RUN_CALIBRATION"
printf '  EVAL_SPLIT=%s\n' "$EVAL_SPLIT"
printf '  EXTRA_EVAL_SPLITS=%s\n' "$EXTRA_EVAL_SPLITS"
printf '  CALIBRATION_SPLITS=%s\n' "$CALIBRATION_SPLITS"

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
    run_train_config "$config"
  done
else
  log "skip train"
fi

if [ "$RUN_EVAL" = "1" ]; then
  for task in $TASKS; do
    ckpt="$(eval_ckpt_for_task "$task")"
    if [ "$task" = "multitask" ]; then
      run_eval_report "$ckpt" "$EVAL_SPLIT" "$MULTITASK_TURN_EVAL_REPORT" turn
      run_eval_report "$ckpt" "$EVAL_SPLIT" "$MULTITASK_RESPONSE_EVAL_REPORT" response
    else
      report="$(eval_report_for_task "$task")"
      run_eval_report "$ckpt" "$EVAL_SPLIT" "$report"
    fi
  done
else
  log "skip eval"
fi

if [ "$RUN_EXTRA_EVAL" = "1" ]; then
  if contains_task response; then
    for split in $EXTRA_EVAL_SPLITS; do
      run_eval_report \
        "$RESPONSE_BEST_RESPONSE_CKPT" \
        "$split" \
        "$REPORT_DIR/${RESPONSE_REPORT_PREFIX}_best_response_${split}_eval.json"
    done
  fi
  if contains_task multitask; then
    for split in $EXTRA_EVAL_SPLITS; do
      run_eval_report \
        "$MULTITASK_BEST_TURN_CKPT" \
        "$split" \
        "$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_best_turn_${split}_eval.json" \
        turn
      run_eval_report \
        "$MULTITASK_BEST_RESPONSE_CKPT" \
        "$split" \
        "$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_best_response_${split}_eval.json" \
        response
    done
  fi
else
  log "skip extra eval"
fi

if [ "$RUN_CALIBRATION" = "1" ]; then
  if contains_task response; then
    for split in $CALIBRATION_SPLITS; do
      run_response_calibration \
        "$RESPONSE_BEST_CKPT" \
        "$split" \
        "$REPORT_DIR/${RESPONSE_REPORT_PREFIX}_best_calibration_${split}.json"
      run_response_calibration \
        "$RESPONSE_BEST_RESPONSE_CKPT" \
        "$split" \
        "$REPORT_DIR/${RESPONSE_REPORT_PREFIX}_best_response_calibration_${split}.json"
    done
  fi
  if contains_task multitask; then
    for split in $CALIBRATION_SPLITS; do
      run_response_calibration \
        "$MULTITASK_BEST_CKPT" \
        "$split" \
        "$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_best_calibration_${split}.json"
      run_response_calibration \
        "$MULTITASK_BEST_RESPONSE_CKPT" \
        "$split" \
        "$REPORT_DIR/${MULTITASK_REPORT_PREFIX}_best_response_calibration_${split}.json"
    done
  fi
else
  log "skip calibration"
fi

log "pipeline finished"
