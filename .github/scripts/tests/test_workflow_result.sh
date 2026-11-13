#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Exercise the actual workflow steps without contacting GitHub or waiting on retries.
set -euo pipefail
cd "$(dirname "$0")/../../.."
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir "$TEST_DIR/bin"

awk '
  /      - name: Get workflow result/ { step = 1; next }
  step && /        run: \|/ { body = 1; next }
  body && /^      - name:/ { exit }
  body { sub(/^          /, ""); print }
' .github/workflows/cicd-main.yml > "$TEST_DIR/result.sh"
awk '
  /      - name: Exit$/ { step = 1; next }
  step && /        run: \|/ { body = 1; next }
  body && /^  [^ ]/ { exit }
  body { sub(/^          /, ""); print }
' .github/workflows/cicd-main.yml > "$TEST_DIR/exit.sh"

cat > "$TEST_DIR/bin/gh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
attempt=$(cat "$MOCK_STATE")
attempt=$((attempt + 1))
printf '%s\n' "$attempt" > "$MOCK_STATE"
if [[ "$MOCK_CASE" == unavailable || ( "$MOCK_CASE" == transient && "$attempt" -lt 3 ) ]]; then
    # A failed later page can leave partial stdout; it must never become a verdict.
    printf '104\t0\t0\n'
    echo 'HTTP 502: Bad Gateway' >&2
    exit 1
fi
[[ $# -eq 5 && "$1" == api && "$2" == --paginate ]]
[[ "$3" == 'repos/test/repo/actions/runs/123/jobs?per_page=20&filter=latest' && "$4" == --jq ]]
printf '%s\n' "$MOCK_COUNTS"
MOCK
cat > "$TEST_DIR/bin/sleep" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && ( "$1" == 2 || "$1" == 4 ) ]]
printf '%s\n' "$1" >> "$MOCK_DELAYS"
MOCK
chmod +x "$TEST_DIR/bin/gh" "$TEST_DIR/bin/sleep"
export PATH="$TEST_DIR/bin:$PATH"
export GITHUB_REPOSITORY=test/repo RUN_ID=123
export MOCK_STATE="$TEST_DIR/calls" MOCK_DELAYS="$TEST_DIR/delays"
export GITHUB_OUTPUT="$TEST_DIR/output"

run_case() {
    local name=$1 expected=$2 calls=$3 exit_status=0
    export MOCK_CASE=$4 MOCK_COUNTS=$5
    printf '0\n' > "$MOCK_STATE"
    : > "$GITHUB_OUTPUT"
    : > "$MOCK_DELAYS"
    bash -e -o pipefail "$TEST_DIR/result.sh" > "$TEST_DIR/log" 2>&1 || exit_status=$?
    if [[ "$expected" == error ]]; then
        if [[ "$exit_status" -eq 0 || -s "$GITHUB_OUTPUT" ]]; then
            echo "FAIL: $name returned a verdict instead of an error" >&2
            cat "$TEST_DIR/log" >&2
            exit 1
        fi
        grep -q '::error::' "$TEST_DIR/log"
    else
        if [[ "$exit_status" -ne 0 || "$(cat "$GITHUB_OUTPUT")" != "code=$expected" ]]; then
            echo "FAIL: $name" >&2
            cat "$TEST_DIR/log" >&2
            exit 1
        fi
    fi
    [[ "$(cat "$MOCK_STATE")" -eq "$calls" ]]
    if [[ "$calls" -eq 3 ]]; then
        [[ "$(cat "$MOCK_DELAYS")" == $'2\n4' ]]
    else
        [[ ! -s "$MOCK_DELAYS" ]]
    fi
    echo "PASS: $name"
}

run_case '104 jobs after two transient errors' success 3 transient $'20\t0\t0\n20\t0\t0\n20\t0\t0\n20\t0\t0\n20\t0\t0\n4\t0\t0'
run_case 'success and skipped jobs' success 1 success $'104\t0\t0'
run_case 'real test failure' failure 1 success $'104\t1\t0'
run_case 'failure on a later page' failure 1 success $'20\t0\t0\n20\t1\t0'
run_case 'cancellation' cancelled 1 success $'104\t0\t1'
run_case 'cancellation on a later page' cancelled 1 success $'20\t0\t0\n20\t0\t1'
run_case 'cancellation takes precedence over failure' cancelled 1 success $'104\t2\t1'
run_case 'bounded retrieval failure discards partial stdout' error 3 unavailable ''
run_case 'empty API output' error 1 success ''
run_case 'empty job list' error 1 success $'0\t0\t0'
run_case 'malformed counts' error 1 success $'104\tinvalid\t0'
run_case 'malformed later page' error 1 success $'20\t0\t0\ninvalid'

for result in success failure cancelled ''; do
    status=0
    RESULT="$result" bash -e -o pipefail "$TEST_DIR/exit.sh" > "$TEST_DIR/log" 2>&1 || status=$?
    if [[ "$result" == success ]]; then
        [[ "$status" -eq 0 ]]
    else
        [[ "$status" -eq 1 ]]
    fi
    if grep -q 'unary operator expected' "$TEST_DIR/log"; then
        echo 'FAIL: empty result caused a shell diagnostic' >&2
        exit 1
    fi
    echo "PASS: final exit with result '${result:-empty}'"
done
