# Sleep app: first working step

This is a privacy-preserving sleep/wake research prototype, not a medical
device and not a sleep-stage classifier. Movement is the primary signal;
heart-rate data can later help identify likely quiet wakefulness but must not
be treated as clinical-grade data.

## Step 1: capture an overnight sensor log

`collect_sensors.py` streams accelerometer samples from sensorfwd's data
socket (`/run/sensord.sock`, requested interval 1 s; the HAL delivers about
every 0.65–1 s) and blocks until sensorfwd sends data. It never polls, so
it never wakes the watch by itself. Heart rate is streamed the same way: every
bpm the sensor hub reports, ~1 Hz after a ~11 s PPG warm-up. Movement and
heart rate are separate timestamped JSON Lines events. A movement event has
this shape:

```json
{"type":"movement","observed_at":"2026-11-17T05:33:27.213Z","monotonic_ns":1667845152669,"accelerometer":{"sensor_timestamp":1667782715,"x":979.904,"y":-49.044,"z":110.532}}
```

and a heart-rate event has this shape (`bpm` is `null` while the PPG warms up
or the watch is off the wrist):

```json
{"type":"heart_rate","observed_at":"2026-11-17T05:45:00.230Z","monotonic_ns":2360801138059,"heart_rate":{"sensor_timestamp":2360799758,"bpm":103,"status":1,"samples":1}}
```

Heart rate is continuous because sparser readings are too coarse for staging:
published wrist staging (Walch et al., SLEEP 2019, Apple Watch) used bpm at
~0.2 Hz with 30 s epochs, and `analyze_sleep.py` needs at least one bpm per
30 s epoch for its 2.5 min variability feature. A 10 min cycle could miss a
whole early-night REM period. `collect_sensors.py OUT DURATION 1 600` still
takes one 15 s median reading per 600 s (`samples` = readings in the median,
0 with `bpm` `null` when none came within 90 s), for when battery matters more
than staging.

`sensor_timestamp` is `CLOCK_BOOTTIME` in microseconds, which keeps counting
during suspend. `observed_at` is the sample's own time, derived from it, so a
batch delivered late still gets its true time. `monotonic_ns` is when the
collector received the event. The recording length, the heart-rate cycle and
the battery check all run on `CLOCK_BOOTTIME`. When the battery falls below
10 % the collector writes a `{"type":"stopped","reason":"battery_floor",...}`
event and exits, so a recording never runs the watch flat.

Until 2026-11-17 the collector polled `xyz` with one `dbus-send` fork per
second and kept the HRM on all night. That cost 53 mA (376 mAh in 7.07 h,
flat at 05:31) against an 8.8 mA wrist baseline.
D-Bus `heartRate` reads were cached values: `sensor_timestamp` never changed
and a reading could be stale. The previous version is
`36e0c29:app-sleep/collect_sensors.py`.

The launcher tile starts a ten-hour recording. It writes:

- JSONL data to `/home/ceres/sleep-recordings/sleep-<UTC timestamp>.jsonl`
- the collector's diagnostic output to a matching `.log` file
- its active PID to `/home/ceres/.local/state/sleep-recorder/recording.pid`

The `sleep-recorder.service` system service prevents launcher shutdown from
stopping the recording. Starting an already active service is a no-op. It
starts `sensorfwd` immediately before the recording and stops it afterwards;
Wi-Fi stays off under `min-power`. It removes the PID file when the recording
ends. Data files are created with owner-only permissions.

## Current user flow

Open **Start Sleep Recording (10h)** in the AsteroidOS launcher before going
to bed. A silent banner confirms that the service is running or reports a
start failure, then clears after four seconds. The display follows its normal
blanking timers. The service continues in the background when the display
blanks. The recording finishes after ten hours
or can be stopped from a shell with:

```sh
echo "" | sudo -S systemctl stop sleep-recorder.service
```

## Retrieving a recording for analysis

Recordings stay on the watch (`/home/ceres/sleep-recordings/`) until pulled
off for analysis. `analyze_sleep.py` expects local `.jsonl` files in this
repository's root. `fetch-sleep-recording.sh` copies the newest recording
there:

```sh
./fetch-sleep-recording.sh [ceres@watch-ip]
```

Defaults to `ceres@192.168.0.101`; pass a host argument if the watch's IP has
changed.

## Next steps

1. Add a status/stop screen and show the active recording duration.
2. Validate movement-derived sleep/wake estimates against a sleep diary.
3. Only then consider a conservative summary (bedtime, wake time, estimated
   total sleep, and awake periods). Do not make medical or sleep-stage claims.
