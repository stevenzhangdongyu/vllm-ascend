#!/usr/bin/env bash
# Run one GDN fwd_h NPU case and compare its outputs directly with a PT reference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 15 || $# -gt 16 ]]; then
    cat <<'EOF' >&2
Usage:
  check_gdn_fwd_h_accuracy.sh \
    B T kH vH D VDim isVariedLen tokenBatch chunkSize \
    useInitialState storeFinalState dtype reference.pt device gDType [stateDType]

Example (two logical sequences packed into batch 1):
  bash tools/check_gdn_fwd_h_accuracy.sh \
    1 8192 4 16 128 128 1 2 64 0 0 bf16 ./fwd_h_ref.pt 0 float fp32
EOF
    exit 2
fi

STATE_DTYPE=${16:-fp32}
REFERENCE_PATH=${13}
if [[ ! -f "$REFERENCE_PATH" ]]; then
    echo "ERROR: reference file not found: $REFERENCE_PATH" >&2
    exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${GDN_ACCURACY_OUTPUT_DIR:-./accuracy_results}/fwd_h_${TIMESTAMP}"
mkdir -p "$REPORT_DIR"

COMMAND=(
    python3 "${SCRIPT_DIR}/test_fwd_h_1.py"
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
    "${10}" "${11}" "${12}" 1 1 "$REFERENCE_PATH" "${14}" "${15}" "$STATE_DTYPE"
    --report-json "$REPORT_DIR/report.json"
)

printf '[INFO] Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
set +e
"${COMMAND[@]}" 2>&1 | tee "$REPORT_DIR/accuracy.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "$STATUS" >"$REPORT_DIR/exit_code"
echo "[INFO] Accuracy artifacts: $REPORT_DIR"
exit "$STATUS"
