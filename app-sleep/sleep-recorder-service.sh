#!/bin/sh
# Main process for the persistent ten-hour sleep-recording systemd service.

set -eu

collector=/home/ceres/collect_sensors.py
recordings=/home/ceres/sleep-recordings
state=/home/ceres/.local/state/sleep-recorder
pid_file="$state/recording.pid"
duration_seconds=36000

mkdir -p "$recordings" "$state"
umask 077

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$recordings/sleep-$timestamp.jsonl"
log="$recordings/sleep-$timestamp.log"

cleanup() {
    rm -f "$pid_file"
}
stop() {
    cleanup
    exit 0
}
trap cleanup EXIT
trap stop INT TERM

echo "$$" > "$pid_file"
# 1 s accelerometer interval, heart rate streamed continuously (0); see
# app-sleep/sleep-app.md for why sparser heart rate is too coarse for staging.
python3 "$collector" "$output" "$duration_seconds" 1 0 >"$log" 2>&1
