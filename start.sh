#!/usr/bin/env bash
set -Eeuo pipefail

# P800/Linux 默认一键流程：
#   ./start.sh
#
# 默认只把原始 XML 放在 /workspace/dataset，其余产物仍写到当前仓库的 data/ 和 outputs/。
# 常用开关：
#   TASKS=turn ./start.sh                         # 只跑 turn 训练和对应 eval
#   TASKS=response ./start.sh                     # 只跑 response 训练和对应 eval
#   RUN_VALIDATE=0 ./start.sh                     # 跳过独立 validate-events 报告
#   RUN_TRAIN=0 RUN_EVAL=0 ./start.sh             # 只生成 preprocess/build/manifest 数据
#   RUN_PREPROCESS=0 RUN_VALIDATE=0 SKIP_BUILD_VALIDATION=1 ./start.sh  # 复用已校验事件数据

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

# 目录约定：RAW_XML_DIR 在训练机数据盘，其余默认都在仓库当前目录下。
RAW_XML_DIR="${RAW_XML_DIR:-/workspace/dataset}"
DATA_ROOT="${DATA_ROOT:-data}"
BRONZE_EVENTS_DIR="${BRONZE_EVENTS_DIR:-$DATA_ROOT/bronze_events}"
SAMPLES_DIR="${SAMPLES_DIR:-$DATA_ROOT/silver_samples}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_ROOT/manifests}"
REPORT_DIR="${REPORT_DIR:-outputs/reports}"
EVENT_VALIDATION_REPORT="${EVENT_VALIDATION_REPORT:-$REPORT_DIR/event_validation.json}"

WORKERS="${WORKERS:-$(default_workers)}"

# TASKS 控制训练/评估哪些任务；默认 turn 和 response 都跑。
# 可选值：turn、response，多个任务用空格分隔，例如 TASKS="turn response"。
TASKS="${TASKS:-turn response}"
TURN_TRAIN_CONFIG="${TURN_TRAIN_CONFIG:-configs/turn_p800.yaml}"
RESPONSE_TRAIN_CONFIG="${RESPONSE_TRAIN_CONFIG:-configs/response_p800.yaml}"

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
TURN_EVAL_REPORT="${TURN_EVAL_REPORT:-$REPORT_DIR/turn_p800_${EVAL_SPLIT}_eval.json}"
RESPONSE_EVAL_REPORT="${RESPONSE_EVAL_REPORT:-$REPORT_DIR/response_p800_${EVAL_SPLIT}_eval.json}"

train_config_for_task() {
  case "$1" in
    turn)
      printf '%s\n' "$TURN_TRAIN_CONFIG"
      ;;
    response)
      printf '%s\n' "$RESPONSE_TRAIN_CONFIG"
      ;;
    *)
      echo "unsupported task in TASKS: $1 (expected turn or response)" >&2
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
    *)
      echo "unsupported task in TASKS: $1 (expected turn or response)" >&2
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
    *)
      echo "unsupported task in TASKS: $1 (expected turn or response)" >&2
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
  # 已经单独跑过 validate-events 时，build-dataset 不再重复回放校验。
  # 如果跳过独立 validate 但仍想在 build 阶段校验，保持 SKIP_BUILD_VALIDATION=0。
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
    report="$(eval_report_for_task "$task")"
    if [ ! -f "$ckpt" ]; then
      echo "missing $task eval checkpoint: $ckpt" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$report")"
    run "$PYTHON_BIN" -m mjai.cli eval \
      --ckpt "$ckpt" \
      --split "$EVAL_SPLIT" \
      --report "$report"
  done
else
  log "skip eval"
fi

log "pipeline finished"
