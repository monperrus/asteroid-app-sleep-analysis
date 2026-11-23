#!/bin/sh
# Fetch the most recent sleep recording from the watch and save it into the
# repository root, where analyze_sleep.py expects to find it.

set -eu

watch_host=${1:-ceres@192.168.0.101}
dest=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

latest=$(ssh "$watch_host" "ls -t /home/ceres/sleep-recordings/*.jsonl | head -1")
# No rsync on the watch (AsteroidOS/busybox), so stream the file over ssh instead.
ssh "$watch_host" "cat '$latest'" > "$dest/$(basename "$latest")"
echo "Saved $(basename "$latest") to $dest/"
