#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_FILE="${SCRIPT_DIR}/.env"
readonly VENV_BIN="${SCRIPT_DIR}/.venv/bin"

analysis_pid=""
ai_pid=""
cleanup_started=0

log() {
  printf '[services] %s\n' "$*"
}

fail() {
  printf '[services] ERROR: %s\n' "$*" >&2
  exit 1
}

resolve_command() {
  local command_name="$1"

  if [[ -x "${VENV_BIN}/${command_name}" ]]; then
    printf '%s\n' "${VENV_BIN}/${command_name}"
    return
  fi

  command -v "${command_name}" || fail \
    "${command_name} is not installed. Activate the AI virtual environment and install the project dependencies."
}

stop_services() {
  if (( cleanup_started )); then
    return
  fi
  cleanup_started=1

  trap - EXIT INT TERM
  log "Stopping services..."

  local pid
  for pid in "${ai_pid}" "${analysis_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  for pid in "${ai_pid}" "${analysis_pid}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done

  log "Both services stopped."
}

trap stop_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -f "${ENV_FILE}" ]] || fail \
  "${ENV_FILE} is missing. Copy .env.example to .env and configure its tokens and provider key."

readonly DOTENV_BIN="$(resolve_command dotenv)"
readonly ANALYSIS_BIN="$(resolve_command code-analysis-api)"
readonly UVICORN_BIN="$(resolve_command uvicorn)"

cd -- "${SCRIPT_DIR}"

log "Starting code analysis on http://127.0.0.1:8000"
PYTHONUNBUFFERED=1 "${DOTENV_BIN}" run -- "${ANALYSIS_BIN}" \
  > >(sed -u 's/^/[analysis] /') \
  2> >(sed -u 's/^/[analysis] /' >&2) &
analysis_pid=$!

log "Starting AI orchestrator on http://127.0.0.1:8010"
PYTHONUNBUFFERED=1 "${DOTENV_BIN}" run -- env \
  PYTHONPATH="${SCRIPT_DIR}/src" \
  AI_SKILL_PROFILE_TIMEOUT_SECONDS="${AI_SKILL_PROFILE_TIMEOUT_SECONDS:-180}" \
  "${UVICORN_BIN}" sharek_agents.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8010 \
  > >(sed -u 's/^/[ai] /') \
  2> >(sed -u 's/^/[ai] /' >&2) &
ai_pid=$!

log "Running: analysis PID ${analysis_pid}, AI PID ${ai_pid}"
log "Press Ctrl+C once to stop both services."

set +e
wait -n "${analysis_pid}" "${ai_pid}"
service_exit_code=$?
set -e

if (( service_exit_code == 0 )); then
  log "A service exited; stopping the remaining service."
else
  log "A service exited with status ${service_exit_code}; stopping the remaining service."
fi

exit "${service_exit_code}"
