# app-sleep — overnight sleep/wake recording

Privacy-preserving sleep/wake research prototype (not a medical device, not a
sleep-stage classifier). Movement is the primary signal; heart-rate data can
later help identify likely quiet wakefulness but must not be treated as
clinical-grade data.

## Contents

| File | Purpose |
|---|---|
| `sleep-app.md` | design + usage record (event format, flow, next steps) |
| `collect_sensors.py` | collector: accelerometer streamed from the sensorfwd socket (~1 Hz) + heart rate streamed (~1 Hz) → JSONL; stops below 10 % battery (no Python D-Bus deps) |
| `test_collect_sensors.py` | host tests for the frame parser and timestamp mapping (`python3 -m pytest -q test_collect_sensors.py`) |
| `sleep-recorder.service` | systemd **system** unit (survives launcher death), runs the recorder as ceres |
| `sleep-recorder-service.sh` | the service's main process (10 h recording, PID file, traps) |
| `sleep-recorder-launcher.sh` | starts the service and shows a brief, silent AsteroidOS banner |
| `sleep-recorder.desktop` | launcher tile — "Start Sleep Recording (10h)", `ios-moon-outline` |
| `fetch-sleep-recording.sh` | host-side: pull newest `.jsonl` into this repository's root, for `analyze_sleep.py` |

## Install on the watch

```
scp collect_sensors.py ceres@192.168.0.101:/home/ceres/
scp sleep-recorder-service.sh ceres@192.168.0.101:/home/ceres/.local/bin/sleep-recorder-service
scp sleep-recorder-launcher.sh ceres@192.168.0.101:/home/ceres/.local/bin/sleep-recorder-launcher
scp sleep-recorder.desktop sleep-recorder.service ceres@192.168.0.101:/home/ceres/
echo "" | sudo -S sh -c '
    cp sleep-recorder.service /etc/systemd/system/ &&
    cp sleep-recorder.desktop /usr/share/applications/ &&
    chmod 700 /home/ceres/.local/bin/sleep-recorder-* &&
    systemctl daemon-reload'
systemctl --user restart asteroid-launcher.service   # from a user session
```

Requires `python3-core python3-json` on the watch.

## Data

- JSONL + diagnostic `.log` per night in `/home/ceres/sleep-recordings/`
- PID file `/home/ceres/.local/state/sleep-recorder/recording.pid`
- Stop early: `echo "" | sudo -S systemctl stop sleep-recorder.service`
- Analysis lives at this repository's root (`analyze_sleep.py`)
