#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd -P)"
SOURCE_REL="output/rl_transfer/cifar10_rtx_phase2_screen"
DATA_REL="data/cifar10"
STUDY_REL="${D1_STUDY_REL:-output/rl_transfer/cifar10_rtx_d1_residual_20260729}"
EXPECTED_SOURCE_SHA="efd96c5775187ac29fbd1453e3d1654d26373fc17b7c0d22b0e4955215a0e054"
EXPECTED_DATA_SHA="c1adf901d7d67ca1df1a1d0d5ae49a079ed82d7ac742568da8041aa60a54f9b7"
BRANCH="agent/d1-residual-ranker"
INNER_DEADLINE_SECONDS=28680
OUTER_TIMEOUT_SECONDS=28795
OUTER_KILL_GRACE_SECONDS=5

if (( OUTER_TIMEOUT_SECONDS - INNER_DEADLINE_SECONDS < 60 ||
      OUTER_TIMEOUT_SECONDS + OUTER_KILL_GRACE_SECONDS > 28800 )); then
  echo "D1 deadline constants violate the hard eight-hour envelope" >&2
  exit 64
fi

if [[ ! "$STUDY_REL" =~ ^output/rl_transfer/[A-Za-z0-9._-]+$ ]]; then
  echo "D1_STUDY_REL must be one safe directory below output/rl_transfer" >&2
  exit 64
fi

SOURCE="$ROOT/$SOURCE_REL"
DATA="$ROOT/$DATA_REL"
STUDY="$ROOT/$STUDY_REL"
CONTROL="${STUDY}_control"
ACTIVE="$CONTROL/active_attempt"

require_safe_control() {
  [[ -d "$CONTROL" && ! -L "$CONTROL" ]] || {
    echo "D1 control directory is missing or is a symlink" >&2
    return 1
  }
  [[ "$(cd "$CONTROL" && pwd -P)" == "$CONTROL" ]] || {
    echo "D1 control directory resolves outside its fixed path" >&2
    return 1
  }
  [[ -z "$(find "$CONTROL" -type l -print -quit)" ]] || {
    echo "D1 control directory cannot contain symlinks" >&2
    return 1
  }
}

validate_attempt() {
  local attempt="$1"
  local parent="${attempt%/*}"
  local name="${attempt##*/}"
  [[ "$parent" == "$CONTROL" ]] || {
    echo "D1 attempt must be inside the fixed control directory" >&2
    return 1
  }
  [[ "$name" =~ ^full-[0-9]{8}T[0-9]{6}Z$ ]] || {
    echo "D1 attempt has an invalid identifier" >&2
    return 1
  }
}

read_active_attempt() {
  require_safe_control
  [[ -f "$ACTIVE" && ! -L "$ACTIVE" ]] || {
    echo "No safe active D1 attempt is recorded" >&2
    return 1
  }
  local attempt
  IFS= read -r attempt < "$ACTIVE"
  validate_attempt "$attempt"
  printf '%s\n' "$attempt"
}

validate_pid() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "D1 control file contains an invalid PID" >&2
    return 1
  }
}

retryable_exit_code() {
  [[ "$1" == "130" || "$1" == "143" ]]
}

atomic_text() {
  local destination="$1"
  local value="$2"
  require_safe_control
  [[ "${destination%/*}" == "$CONTROL" && ! -L "$destination" ]] || {
    echo "D1 control write escaped its fixed directory" >&2
    return 1
  }
  local temporary="${destination}.tmp.$$"
  [[ ! -e "$temporary" && ! -L "$temporary" ]] || {
    echo "D1 atomic temporary path already exists" >&2
    return 1
  }
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$destination"
}

current_sha() {
  git -C "$ROOT" rev-parse HEAD
}

require_clean_feature_branch() {
  [[ "$(git -C "$ROOT" branch --show-current)" == "$BRANCH" ]] || {
    echo "D1 must run from $BRANCH" >&2
    return 1
  }
  git -C "$ROOT" diff --quiet
  git -C "$ROOT" diff --cached --quiet
  [[ -z "$(git -C "$ROOT" ls-files --others --exclude-standard)" ]] || {
    echo "D1 requires a clean tracked and untracked worktree" >&2
    return 1
  }
}

require_no_other_compute() {
  local compute
  compute="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "$compute" ]] || {
    echo "Another CUDA compute process is active:" >&2
    printf '%s\n' "$compute" >&2
    return 1
  }
}

current_input_binding() {
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python - \
    "$SOURCE" "$DATA" "$EXPECTED_SOURCE_SHA" "$EXPECTED_DATA_SHA" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

import torch
import torchvision

from rl_transfer.reproducibility import tree_digest

source = Path(sys.argv[1])
data = Path(sys.argv[2])
expected_source = sys.argv[3]
expected_data = sys.argv[4]
manifest = source / "screen_manifest.json"
manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
data_digest = tree_digest(data)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if manifest_digest != expected_source:
    raise SystemExit("source manifest digest changed")
if data_digest != expected_data:
    raise SystemExit("CIFAR-10 tree digest changed")
print(
    json.dumps(
        {
            "data_tree_sha256": data_digest,
            "gpu_name": torch.cuda.get_device_name(0),
            "python_version": sys.version.split()[0],
            "source_manifest_sha256": manifest_digest,
            "source_tree_sha256": tree_digest(source),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

preflight() {
  cd "$ROOT"
  mkdir -p "$CONTROL"
  require_safe_control
  require_clean_feature_branch
  command -v timeout >/dev/null
  command -v nvidia-smi >/dev/null
  command -v sha256sum >/dev/null
  command -v tar >/dev/null
  command -v find >/dev/null
  command -v cmp >/dev/null
  [[ -x .venv/bin/python ]]
  [[ -f "$SOURCE/screen_manifest.json" && \
    ! -L "$SOURCE/screen_manifest.json" ]]
  [[ -f "$SOURCE/screen_manifest.json.sha256" && \
    ! -L "$SOURCE/screen_manifest.json.sha256" ]]
  [[ -d "$SOURCE" && ! -L "$SOURCE" ]]
  [[ -d "$DATA" && ! -L "$DATA" ]]
  [[ -z "$(find "$SOURCE" "$DATA" -type l -print -quit)" ]]
  [[ "$(sha256sum "$SOURCE/screen_manifest.json" | awk '{print $1}')" == \
    "$EXPECTED_SOURCE_SHA" ]]
  [[ "$(tr -d '[:space:]' < "$SOURCE/screen_manifest.json.sha256")" == \
    "$EXPECTED_SOURCE_SHA" ]]
  local available_kib
  available_kib="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
  (( available_kib >= 5242880 )) || {
    echo "D1 requires at least 5 GiB of free disk space" >&2
    return 1
  }
  require_no_other_compute
  nvidia-smi
  atomic_text "$CONTROL/input_binding.json" "$(current_input_binding)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    -m rl_transfer.phase2_residual_d1_cli \
    --source-manifest "$SOURCE/screen_manifest.json" \
    --source-root "$SOURCE" \
    --output-dir "$STUDY" \
    --data-root "$DATA" \
    --deadline-seconds "$INNER_DEADLINE_SECONDS" \
    --dry-run > "$CONTROL/dry-run.json"
  atomic_text "$CONTROL/code_commit.txt" "$(current_sha)"
  local freeze_temporary="$CONTROL/pip_freeze.txt.tmp.$$"
  [[ ! -e "$freeze_temporary" && ! -L "$freeze_temporary" ]]
  .venv/bin/python -m pip freeze > "$freeze_temporary"
  mv "$freeze_temporary" "$CONTROL/pip_freeze.txt"
  sha256sum "$CONTROL/pip_freeze.txt" > "$CONTROL/pip_freeze.txt.sha256"
  echo "preflight_ok study=$STUDY control=$CONTROL"
}

smoke() {
  preflight
  local stamp smoke_root smoke_log rc
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  smoke_root="${STUDY}_smoke_${stamp}"
  smoke_log="$CONTROL/smoke-${stamp}.log"
  [[ ! -e "$smoke_root" ]]
  set +e
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 420s \
    .venv/bin/python -u -m rl_transfer.phase2_residual_d1_cli \
    --source-manifest "$SOURCE/screen_manifest.json" \
    --source-root "$SOURCE" \
    --output-dir "$smoke_root" \
    --data-root "$DATA" \
    --deadline-seconds 300 \
    --smoke-test > "$smoke_log" 2>&1
  rc=$?
  set -e
  (( rc == 0 )) || {
    echo "D1 smoke failed with exit $rc; see $smoke_log" >&2
    return "$rc"
  }
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python - "$smoke_root" <<'PY'
from pathlib import Path
import sys

from rl_transfer.artifacts import sha256_file
from rl_transfer.verified_artifacts import load_verified_json

root = Path(sys.argv[1])
manifest = load_verified_json(root / "smoke_manifest.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


require(manifest.get("status") == "complete", "smoke did not complete")
require(manifest.get("target_calls") == 0, "smoke used target calls")
require(manifest.get("hidden_target_calls") == 0, "smoke used hidden targets")
require(
    manifest.get("target_evaluation_performed") is False,
    "smoke performed target evaluation",
)
require(
    manifest.get("hidden_target_evaluation_performed") is False,
    "smoke performed hidden-target evaluation",
)
require(
    manifest.get("authorizes_d1_promotion") is False,
    "smoke cannot authorize promotion",
)
require(
    sha256_file(root / "smoke_residual_ranker.pt")
    == manifest.get("checkpoint_sha256"),
    "smoke checkpoint digest changed",
)
require(
    sha256_file(root / "smoke_results.jsonl")
    == manifest.get("results_sha256"),
    "smoke results digest changed",
)
require(
    sha256_file(root / "smoke_query_traces.jsonl")
    == manifest.get("query_traces_sha256"),
    "smoke trace digest changed",
)
print("D1 GPU smoke verified")
PY
  local binding_sha
  binding_sha="$(sha256sum "$CONTROL/input_binding.json" | awk '{print $1}')"
  atomic_text "$CONTROL/smoke.ok" "$(current_sha):${binding_sha}"
  atomic_text "$CONTROL/latest_smoke" "$smoke_root"
  echo "smoke_ok output=$smoke_root log=$smoke_log"
}

validate_smoke_marker() {
  require_safe_control
  [[ -f "$CONTROL/smoke.ok" && ! -L "$CONTROL/smoke.ok" ]]
  [[ -f "$CONTROL/input_binding.json" && \
    ! -L "$CONTROL/input_binding.json" ]]
  local binding_sha expected actual
  binding_sha="$(sha256sum "$CONTROL/input_binding.json" | awk '{print $1}')"
  expected="$(current_sha):${binding_sha}"
  IFS= read -r actual < "$CONTROL/smoke.ok"
  [[ "$actual" == "$expected" ]] || {
    echo "The smoke test does not match current code and input bytes" >&2
    return 1
  }
}

run_cli_once() {
  local log="$1"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
    -m rl_transfer.phase2_residual_d1_cli \
    --source-manifest "$SOURCE/screen_manifest.json" \
    --source-root "$SOURCE" \
    --output-dir "$STUDY" \
    --data-root "$DATA" \
    --deadline-seconds "$INNER_DEADLINE_SECONDS" >> "$log" 2>&1 &
  local child=$!
  trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
  set +e
  wait "$child"
  local rc=$?
  set -e
  trap - TERM INT
  return "$rc"
}

safe_d1b_retry_available() {
  local prior_rc="$1"
  retryable_exit_code "$prior_rc"
  [[ -f "$STUDY/d1a/d1_manifest.json" ]]
  compgen -G "$STUDY/d1b/ppo_block_*.receipt.json" >/dev/null
  .venv/bin/python - "$STUDY" <<'PY'
from pathlib import Path
import sys
import time

from rl_transfer.phase2_residual_d1 import validate_source_only_payload
from rl_transfer.verified_artifacts import load_verified_json

root = Path(sys.argv[1])
study = load_verified_json(root / "study_manifest.json")
d1a = load_verified_json(root / "d1a" / "d1_manifest.json")
d1b = load_verified_json(root / "d1b" / "d1b_manifest.json")
remaining = float(study["deadline_epoch_seconds"]) - time.time()
validate_source_only_payload(study, "retry study manifest")
validate_source_only_payload(d1a, "retry D1a manifest")
validate_source_only_payload(d1b, "retry D1b manifest")
if study.get("status") != "running":
    raise SystemExit("only an interrupted running study may retry")
if d1a.get("status") != "complete":
    raise SystemExit("retry requires a complete D1a")
if d1b.get("status") != "running":
    raise SystemExit("only an interrupted running D1b may retry")
if remaining <= 120:
    raise SystemExit("too little persisted deadline remains for a retry")
final_names = (
    "residual_ranker_ppo.pt",
    "source_results.jsonl",
    "source_query_traces.jsonl",
    "asr_by_query.svg",
    "final_asr.svg",
)
if any((root / "d1b" / name).exists() for name in final_names):
    raise SystemExit("partial final D1b evidence cannot be retried")
receipts = sorted((root / "d1b").glob("ppo_block_*.receipt.json"))
if not 1 <= len(receipts) <= 4:
    raise SystemExit("retry requires a bounded committed receipt prefix")
if any(
    not path.with_suffix(path.suffix + ".sha256").is_file()
    for path in receipts
):
    raise SystemExit("retry receipt sidecar is incomplete")
PY
}

attempt_loop() {
  local attempt="$1"
  require_safe_control
  validate_attempt "$attempt"
  local log="${attempt}.log"
  cd "$ROOT"
  echo "D1 attempt 1 started $(date -u +%FT%TZ)" >> "$log"
  local rc=0
  run_cli_once "$log" || rc=$?
  if (( rc != 0 )) && safe_d1b_retry_available "$rc"; then
    echo "D1 safe D1b resume attempt 2 started $(date -u +%FT%TZ)" >> "$log"
    rc=0
    run_cli_once "$log" || rc=$?
  fi
  return "$rc"
}

telemetry() {
  local parent_pid="$1"
  local attempt="$2"
  require_safe_control
  validate_pid "$parent_pid"
  validate_attempt "$attempt"
  local gpu="${attempt}.gpu.csv"
  local heartbeat="${attempt}.heartbeat"
  echo "timestamp,index,name,utilization_gpu,memory_used,memory_total,temperature,power_draw" \
    > "$gpu"
  while kill -0 "$parent_pid" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >> "$gpu" 2>/dev/null || true
    .venv/bin/python - "$STUDY" "$parent_pid" > "${heartbeat}.tmp.$$" <<'PY' || true
from pathlib import Path
import sys
import time

from rl_transfer.verified_artifacts import load_verified_json

root = Path(sys.argv[1])
print("epoch_seconds", time.time())
print("launcher_pid", sys.argv[2])
for relative in (
    "study_manifest.json",
    "d1a/d1_manifest.json",
    "d1b/d1b_manifest.json",
):
    path = root / relative
    if path.is_file():
        try:
            value = load_verified_json(path)
            print(relative, value.get("status"), value.get("study_outcome"))
            if relative == "study_manifest.json":
                print(
                    "remaining_seconds",
                    float(value["deadline_epoch_seconds"]) - time.time(),
                )
        except Exception as error:
            print(relative, "transient", type(error).__name__)
print(
    "d1b_blocks",
    len(tuple((root / "d1b").glob("ppo_block_*.receipt.json"))),
)
PY
    [[ -f "${heartbeat}.tmp.$$" ]] && mv "${heartbeat}.tmp.$$" "$heartbeat"
    sleep 30
  done
}

worker() {
  local attempt="$1"
  require_safe_control
  validate_attempt "$attempt"
  cd "$ROOT"
  telemetry "$$" "$attempt" &
  local telemetry_pid=$!
  trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
  local rc=0
  timeout --signal=TERM --kill-after="${OUTER_KILL_GRACE_SECONDS}s" \
    "${OUTER_TIMEOUT_SECONDS}s" \
    "$SCRIPT" _attempt_loop "$attempt" || rc=$?
  atomic_text "${attempt}.exit" "$rc"
  return "$rc"
}

launch() {
  preflight
  validate_smoke_marker
  [[ ! -e "$STUDY" ]] || {
    echo "D1 production study path must be fresh at initial launch" >&2
    return 1
  }
  if [[ -f "$ACTIVE" ]]; then
    local previous previous_pid
    previous="$(read_active_attempt)"
    [[ -f "${previous}.pid" && ! -L "${previous}.pid" ]]
    IFS= read -r previous_pid < "${previous}.pid"
    validate_pid "$previous_pid"
    if kill -0 "$previous_pid" 2>/dev/null; then
      echo "A D1 launcher is already active: $previous" >&2
      return 1
    fi
  fi
  local stamp attempt pid
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  attempt="$CONTROL/full-${stamp}"
  validate_attempt "$attempt"
  [[ ! -e "${attempt}.nohup" && ! -e "${attempt}.pid" ]]
  nohup "$SCRIPT" _worker "$attempt" > "${attempt}.nohup" 2>&1 < /dev/null &
  pid=$!
  atomic_text "${attempt}.pid" "$pid"
  atomic_text "$ACTIVE" "$attempt"
  echo "launched pid=$pid attempt=$attempt study=$STUDY"
}

status() {
  local attempt pid
  attempt="$(read_active_attempt)"
  [[ -f "${attempt}.pid" && ! -L "${attempt}.pid" ]]
  IFS= read -r pid < "${attempt}.pid"
  validate_pid "$pid"
  ps -p "$pid" -o pid,ppid,lstart,etime,state,args || true
  [[ -f "${attempt}.exit" ]] && echo "exit_code=$(cat "${attempt}.exit")"
  [[ -f "${attempt}.heartbeat" ]] && tail -n 20 "${attempt}.heartbeat"
  [[ -f "${attempt}.log" ]] && tail -n 60 "${attempt}.log"
  nvidia-smi
}

run_verifier() {
  local archive="${1:-}"
  local checksums="${2:-}"
  cd "$ROOT"
  require_clean_feature_branch
  local attempt output temporary
  attempt="$(read_active_attempt)"
  [[ -f "${attempt}.exit" && ! -L "${attempt}.exit" ]] || {
    echo "D1 is not terminal yet" >&2
    return 1
  }
  local exit_code recorded_commit recorded_binding freeze_temporary
  IFS= read -r exit_code < "${attempt}.exit"
  [[ "$exit_code" =~ ^[0-9]+$ ]] || {
    echo "D1 terminal exit code is malformed" >&2
    return 1
  }
  IFS= read -r recorded_commit < "$CONTROL/code_commit.txt"
  [[ "$recorded_commit" == "$(current_sha)" ]] || {
    echo "D1 verification commit differs from preflight" >&2
    return 1
  }
  recorded_binding="$(cat "$CONTROL/input_binding.json")"
  [[ "$recorded_binding" == "$(current_input_binding)" ]] || {
    echo "D1 source, data, GPU, or runtime binding changed" >&2
    return 1
  }
  sha256sum -c "$CONTROL/pip_freeze.txt.sha256"
  freeze_temporary="$CONTROL/pip_freeze.verify.tmp.$$"
  [[ ! -e "$freeze_temporary" && ! -L "$freeze_temporary" ]]
  .venv/bin/python -m pip freeze > "$freeze_temporary"
  cmp "$CONTROL/pip_freeze.txt" "$freeze_temporary"
  rm "$freeze_temporary"
  local -a package_arguments=()
  if [[ -n "$archive" || -n "$checksums" ]]; then
    [[ -n "$archive" && -n "$checksums" ]] || {
      echo "D1 archive and checksum paths must be supplied together" >&2
      return 1
    }
    package_arguments=(--archive "$archive" --checksums "$checksums")
    output="${attempt}.package-verification.json"
  else
    output="${attempt}.verification.json"
  fi
  temporary="${output}.tmp.$$"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    -m rl_transfer.phase2_residual_d1_verify \
    --study-root "$STUDY" \
    --source-manifest "$SOURCE/screen_manifest.json" \
    --source-root "$SOURCE" \
    --data-root "$DATA" \
    "${package_arguments[@]}" > "$temporary"
  mv "$temporary" "$output"
  sha256sum "$output" > "${output}.sha256"
  echo "verified=$output"
}

verify() {
  run_verifier
}

package_study() {
  local base sums archive stamp
  base="$(basename "$STUDY")"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  sums="$CONTROL/${base}-${stamp}.SHA256SUMS"
  archive="$CONTROL/${base}-${stamp}.tar.gz"
  run_verifier "$archive" "$sums"
  (
    cd "$CONTROL"
    sha256sum -c "$(basename "${archive}.sha256")"
  )
  echo "archive=$archive"
  echo "artifact_checksums=$sums"
}

usage() {
  echo "usage: scripts/d1_remote.sh {preflight|smoke|launch|status|verify|package}" >&2
}

case "${1:-}" in
  preflight) preflight ;;
  smoke) smoke ;;
  launch) launch ;;
  status) status ;;
  verify) verify ;;
  package) package_study ;;
  _attempt_loop) attempt_loop "$2" ;;
  _telemetry) telemetry "$2" "$3" ;;
  _worker) worker "$2" ;;
  _validate_attempt) validate_attempt "$2" ;;
  _retryable_exit) retryable_exit_code "$2" ;;
  *) usage; exit 64 ;;
esac
