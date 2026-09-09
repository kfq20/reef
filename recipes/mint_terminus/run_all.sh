#!/usr/bin/env bash
set -Eeuo pipefail

# One-command interactive smoke run for the MinT + Terminus + Reef pipeline.
# Run from anywhere:
#   MINT_API_KEY=... E2B_API_KEY=... recipes/mint_terminus/run_all.sh
#
# The script starts only services that are not already listening, prints the
# driver output to the terminal, and keeps a copy of every log under work/.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RECIPE_ROOT="$ROOT/recipes/mint_terminus"
WORK_DIR="${REEF_MINT_WORK_DIR:-$RECIPE_ROOT/work}"
PYTHON="${REEF_MINT_PYTHON:-$ROOT/.venv-tb/bin/python3}"
SHIM_PORT="${REEF_MINT_SHIM_PORT:-8971}"
REEF_PORT="${REEF_MINT_REEF_PORT:-8912}"
UPSTREAM_URL="${MINT_UPSTREAM_URL:-https://mintcn.macaron.xin}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found: $PYTHON" >&2
    echo "Set REEF_MINT_PYTHON to the Python executable that has reef, httpx, and reef-eval installed." >&2
    exit 2
fi
if [[ -z "${MINT_API_KEY:-}" ]]; then
    echo "ERROR: MINT_API_KEY is required." >&2
    exit 2
fi
if [[ -z "${E2B_API_KEY:-}" ]]; then
    echo "ERROR: E2B_API_KEY is required for Terminus episodes." >&2
    exit 2
fi
for task in \
    /root/.reef-eval-tasks/continual-learning/terminal-bench/fix-git \
    /root/.reef-eval-tasks/continual-learning/terminal-bench/cobol-modernization
do
    if [[ ! -f "$task/instruction.md" ]]; then
        echo "ERROR: Terminal-Bench task is missing: $task" >&2
        exit 2
    fi
done

mkdir -p "$WORK_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export REEF_RECIPE_CONFIG_DIR="$RECIPE_ROOT"
export REEF_PROPOSER_TIMEOUT_S="${REEF_PROPOSER_TIMEOUT_S:-600}"
export REEF_PROPOSER_MAX_TOKENS="${REEF_PROPOSER_MAX_TOKENS:-8192}"
export MINT_API_KEY
export E2B_API_KEY
export E2B_API_URL="${E2B_API_URL:?set E2B_API_URL (e.g. http://<e2b-host>:<port>)}"
# Episodes resolve the terminus wrapper (bin/reef-terminus-mint) through the
# serving process's PATH, so bin/ must be on it before reef serve starts.
export PATH="$RECIPE_ROOT/bin:$PATH"

REEF_PID=""
SHIM_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$REEF_PID" ]] && kill -0 "$REEF_PID" 2>/dev/null; then
        echo "[cleanup] stopping Reef (pid $REEF_PID)" >&2
        kill "$REEF_PID" 2>/dev/null || true
        wait "$REEF_PID" 2>/dev/null || true
    fi
    if [[ -n "$SHIM_PID" ]] && kill -0 "$SHIM_PID" 2>/dev/null; then
        echo "[cleanup] stopping OpenAI shim (pid $SHIM_PID)" >&2
        kill "$SHIM_PID" 2>/dev/null || true
        wait "$SHIM_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

port_is_listening() {
    local port=$1
    "$PYTHON" - "$port" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

echo "== Reef MinT/Terminus pipeline =="
echo "repo:       $ROOT"
echo "python:     $PYTHON"
echo "upstream:   $UPSTREAM_URL"
echo "reef URL:   http://127.0.0.1:$REEF_PORT"
echo "shim URL:   http://127.0.0.1:$SHIM_PORT"
echo "logs:       $WORK_DIR"
echo

if port_is_listening "$SHIM_PORT"; then
    echo "[1/4] OpenAI shim already listening on :$SHIM_PORT; reusing it"
else
    echo "[1/4] starting OpenAI shim"
    "$PYTHON" -m recipes.mint_terminus.harness.openai_shim "$SHIM_PORT" "$UPSTREAM_URL" \
        >"$WORK_DIR/openai-shim.log" 2>&1 &
    SHIM_PID=$!
    sleep 1
    if ! kill -0 "$SHIM_PID" 2>/dev/null; then
        echo "ERROR: OpenAI shim exited; see $WORK_DIR/openai-shim.log" >&2
        exit 1
    fi
fi

if curl -sf --max-time 3 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
    echo "[2/4] Reef already healthy on :$REEF_PORT; reusing it"
else
    echo "[2/4] starting Reef"
    "$PYTHON" -m reef serve -c "$RECIPE_ROOT/deployment.yaml" \
        >"$WORK_DIR/reef-serve.log" 2>&1 &
    REEF_PID=$!
    ready=0
    for _ in $(seq 1 120); do
        if curl -sf --max-time 2 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$REEF_PID" 2>/dev/null; then
            echo "ERROR: Reef exited during startup; see $WORK_DIR/reef-serve.log" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "$ready" != 1 ]]; then
        echo "ERROR: Reef did not become healthy within 120 seconds; see $WORK_DIR/reef-serve.log" >&2
        exit 1
    fi
fi

echo "[3/4] running record -> report -> evolve -> poll"
SCENARIO="${REEF_MINT_SCENARIO:-mint-terminus-$(date -u +%Y%m%d-%H%M%S)}"
export REEF_MINT_SERVICE_URL="http://127.0.0.1:$REEF_PORT"
export REEF_MINT_SCENARIO="$SCENARIO"
export REEF_MINT_TASK="${REEF_MINT_TASK:-/root/.reef-eval-tasks/continual-learning/terminal-bench/fix-git}"
echo "      scenario: $SCENARIO"
echo "      task:     $REEF_MINT_TASK"
set +e
"$PYTHON" -m recipes.mint_terminus.run_loop 2>&1 | tee "$WORK_DIR/run-$SCENARIO.log"
driver_status=${PIPESTATUS[0]}
set -e

echo
echo "[4/4] final status"
curl -sS --max-time 10 "http://127.0.0.1:$REEF_PORT/reef/status" \
    -H "Authorization: Bearer reef-local" \
    -H "x-reef-scenario: $SCENARIO" | "$PYTHON" -m json.tool 2>/dev/null | sed -n '1,220p' || true
echo
echo "driver exit code: $driver_status"
echo "service log:      $WORK_DIR/reef-serve.log"
echo "shim log:         $WORK_DIR/openai-shim.log"
echo "driver log:       $WORK_DIR/run-$SCENARIO.log"
exit "$driver_status"
