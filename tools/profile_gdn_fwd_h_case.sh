#!/usr/bin/env bash
# Profile one GDN fwd_h case with msprof op.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_NAME="chunk_gated_delta_rule_fwd_kernel_h_blockdim64_mix_aic"
LAUNCH_COUNT=1
OUTPUT_ROOT="./prof"
TEST_SCRIPT="${SCRIPT_DIR}/test_fwd_h_1.py"
PYTHON_BIN="python3"

usage() {
    cat <<'EOF'
Usage:
  profile_gdn_fwd_h_case.sh [OPTIONS] -- \
    B T kH vH D VDim isVariedLen tokenBatch chunkSize useInitialState storeFinalState \
    dtype useActualInput useActualOutput dataPath device gDType stateDType

Options:
  --test-script=PATH   Python test driver (default: tools/test_fwd_h_1.py)
  --output=DIR         Profile output root (default: ./prof)
  --kernel-name=NAME   Kernel selected by msprof
  --launch-count=N     Number of selected launches (default: 1)
  --python=COMMAND     Python executable (default: python3)
  -h, --help           Show this help

Example:
  bash tools/profile_gdn_fwd_h_case.sh \
    --test-script=./test_fwd_h_1.py --output=./prof -- \
    1 1024 16 32 128 128 0 1 64 0 0 bf16 1 0 ./input.pt 0 float fp32
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-script=*) TEST_SCRIPT="${1#*=}" ;;
        --output=*) OUTPUT_ROOT="${1#*=}" ;;
        --kernel-name=*) KERNEL_NAME="${1#*=}" ;;
        --launch-count=*) LAUNCH_COUNT="${1#*=}" ;;
        --python=*) PYTHON_BIN="${1#*=}" ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

if [[ $# -ne 18 ]]; then
    echo "ERROR: expected 18 case arguments after --, got $#" >&2
    usage
    exit 2
fi

if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: msprof not found; source the CANN environment first" >&2
    exit 127
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 127
fi
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "ERROR: test script not found: $TEST_SCRIPT" >&2
    exit 2
fi
if [[ ! "$LAUNCH_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: launch-count must be a positive integer" >&2
    exit 2
fi

B=$1
T=$2
KH=$3
VH=$4
D=$5
VDIM=$6
IS_VARLEN=$7
TOKEN_BATCH=$8
CHUNK_SIZE=$9
USE_INITIAL_STATE=${10}
STORE_FINAL_STATE=${11}
DTYPE=${12}
USE_ACTUAL_INPUT=${13}
USE_ACTUAL_OUTPUT=${14}
DATA_PATH=${15}
DEVICE=${16}
G_DTYPE=${17}
STATE_DTYPE=${18}

for binary_flag in "$IS_VARLEN" "$USE_INITIAL_STATE" "$STORE_FINAL_STATE" "$USE_ACTUAL_INPUT" "$USE_ACTUAL_OUTPUT"; do
    if [[ "$binary_flag" != "0" && "$binary_flag" != "1" ]]; then
        echo "ERROR: binary flags must be 0 or 1" >&2
        exit 2
    fi
done
if [[ "$USE_ACTUAL_INPUT" == "1" && ! -f "$DATA_PATH" ]]; then
    echo "ERROR: actual input file not found: $DATA_PATH" >&2
    exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CASE_NAME="b${B}_t${T}_kh${KH}_vh${VH}_d${D}_vd${VDIM}_c${CHUNK_SIZE}_${DTYPE}"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${CASE_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

APP_CMD=(
    "$PYTHON_BIN" "$TEST_SCRIPT"
    "$B" "$T" "$KH" "$VH" "$D" "$VDIM" "$IS_VARLEN" "$TOKEN_BATCH" "$CHUNK_SIZE"
    "$USE_INITIAL_STATE" "$STORE_FINAL_STATE" "$DTYPE" "$USE_ACTUAL_INPUT"
    "$USE_ACTUAL_OUTPUT" "$DATA_PATH" "$DEVICE" "$G_DTYPE" "$STATE_DTYPE"
)
MSPROF_CMD=(
    msprof op
    "--kernel-name=$KERNEL_NAME"
    "--launch-count=$LAUNCH_COUNT"
    "--output=$OUTPUT_DIR"
    "${APP_CMD[@]}"
)

{
    echo "kernel_name=$KERNEL_NAME"
    echo "launch_count=$LAUNCH_COUNT"
    echo "test_script=$(realpath "$TEST_SCRIPT")"
    printf "command="
    printf "%q " "${MSPROF_CMD[@]}"
    printf "\n"
} >"$OUTPUT_DIR/case.txt"

echo "[INFO] Case   : $CASE_NAME"
echo "[INFO] Kernel : $KERNEL_NAME"
echo "[INFO] Output : $OUTPUT_DIR"
printf "[INFO] Command: "
printf "%q " "${MSPROF_CMD[@]}"
printf "\n"

set +e
"${MSPROF_CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/msprof.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "$STATUS" >"$OUTPUT_DIR/exit_code"

if [[ $STATUS -ne 0 ]]; then
    echo "[ERROR] Profiling failed with exit code $STATUS; see $OUTPUT_DIR/msprof.log" >&2
    exit "$STATUS"
fi

echo "[INFO] Profiling completed: $OUTPUT_DIR"
