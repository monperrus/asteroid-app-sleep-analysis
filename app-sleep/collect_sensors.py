#!/usr/bin/env python3
"""Record AsteroidOS accelerometer and heart-rate samples as JSON Lines.

Accelerometer samples are streamed from sensorfwd's data socket: the process
blocks until sensorfwd delivers, so it never wakes the device on its own and
forks nothing per sample. Heart rate is streamed the same way, every sample
the hub reports (~1 Hz), unless HEART_RATE_PERIOD > 0 asks for one short
reading per period instead. Control calls go through the
installed `dbus-send` command, keeping the runtime dependency-free beyond
Python's standard library and the `python3-json` package.

Every deadline uses CLOCK_BOOTTIME, which keeps counting while the watch is
suspended; CLOCK_MONOTONIC (time.sleep, select timeouts) does not.
"""

import json
import os
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import time

SERVICE = "com.nokia.SensorService"
MANAGER_PATH = "/SensorManager"
DATA_SOCKET = "/run/sensord.sock"
ACCELEROMETER = "accelerometersensor"
HEART_RATE = "hrmsensor"
BATTERY_CAPACITY = "/sys/class/power_supply/battery/capacity"

# sensorfwd frame: uint32 count, then count TimedXyzData records
# (quint64 timestamp in CLOCK_BOOTTIME microseconds, float x, y, z, padding).
FRAME_HEADER = struct.Struct("<I")
XYZ_RECORD = struct.Struct("<Qfff4x")
# HeartRateData: quint64 timestamp (CLOCK_BOOTTIME us), int32 status, int32 bpm.
HEART_RATE_RECORD = struct.Struct("<Qii")

HEART_RATE_WINDOW = 90.0      # seconds the HRM may take to report a bpm
HEART_RATE_COLLECT = 15.0     # seconds of valid readings per cycle (median)
BATTERY_FLOOR = 10            # percent; below it the recording stops
BATTERY_CHECK_EVERY = 300.0   # seconds
SELECT_TIMEOUT = 30.0         # bounds latency while awake; pauses in suspend


def usage():
    print(
        "Usage: collect_sensors.py OUTPUT.jsonl DURATION_SECONDS "
        "[MOVEMENT_INTERVAL_SECONDS] [HEART_RATE_PERIOD_SECONDS]\n"
        "HEART_RATE_PERIOD_SECONDS 0 (default) streams heart rate continuously.",
        file=sys.stderr,
    )


def boottime():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def dbus(path, method, *arguments):
    """Call a system-bus method and return dbus-send's textual reply."""
    command = [
        "dbus-send",
        "--system",
        "--print-reply",
        f"--dest={SERVICE}",
        path,
        method,
        *arguments,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "unknown D-Bus error"
        raise RuntimeError(f"{' '.join(command)}: {message}")
    return result.stdout


def request_session(sensor, pid):
    dbus(MANAGER_PATH, "local.SensorManager.loadPlugin", f"string:{sensor}")
    reply = dbus(
        MANAGER_PATH,
        "local.SensorManager.requestSensor",
        f"string:{sensor}",
        f"int64:{pid}",
    )
    match = re.search(r"\bint32\s+(-?\d+)", reply)
    if not match:
        raise RuntimeError(f"could not obtain a {sensor} session: {reply.strip()}")
    return int(match.group(1))


def stop_and_release(sensor, interface, session, pid):
    if session is None:
        return
    path = f"{MANAGER_PATH}/{sensor}"
    try:
        dbus(path, f"local.{interface}.stop", f"int32:{session}")
    except RuntimeError:
        pass
    try:
        dbus(
            MANAGER_PATH,
            "local.SensorManager.releaseSensor",
            f"string:{sensor}",
            f"int32:{session}",
            f"int64:{pid}",
        )
    except RuntimeError:
        # Do not hide a successful recording because cleanup raced with
        # sensorfwd shutting down a session.
        pass


def connect_data_socket(session):
    """Open sensorfwd's data socket and bind it to a session."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(DATA_SOCKET)
    greeting = sock.recv(1)
    if greeting != b"\n":
        raise RuntimeError(f"unexpected sensorfwd greeting: {greeting!r}")
    sock.sendall(struct.pack("<i", session))
    return sock


class FrameParser:
    """Reassemble sensorfwd frames from a byte stream."""

    def __init__(self, record=XYZ_RECORD):
        self.record = record
        self.buffer = b""

    def feed(self, data):
        """Return the (timestamp_us, x, y, z) records completed by data."""
        self.buffer += data
        records = []
        while len(self.buffer) >= FRAME_HEADER.size:
            (count,) = FRAME_HEADER.unpack_from(self.buffer)
            size = FRAME_HEADER.size + count * self.record.size
            if len(self.buffer) < size:
                break
            for index in range(count):
                offset = FRAME_HEADER.size + index * self.record.size
                records.append(self.record.unpack_from(self.buffer, offset))
            self.buffer = self.buffer[size:]
        return records


def utc_iso(epoch):
    milliseconds = int((epoch % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + f".{milliseconds:03d}Z"


def sample_wall_time(timestamp_us, wall_now, boottime_now):
    """Wall-clock time of a sample stamped in CLOCK_BOOTTIME microseconds."""
    return wall_now - (boottime_now - timestamp_us / 1e6)


def write_event(output, event_type, payload, observed_epoch=None):
    output.write(
        json.dumps(
            {
                "type": event_type,
                "observed_at": utc_iso(time.time() if observed_epoch is None else observed_epoch),
                "monotonic_ns": time.monotonic_ns(),
                **payload,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def battery_capacity():
    try:
        with open(BATTERY_CAPACITY, encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


class HeartRateCycle:
    """Stream the HRM continuously (period 0), or switch it on every period,
    take one median reading and switch it off.

    Readings come from the data socket: a D-Bus `heartRate` read returns the
    last cached value (stale across sessions, `sensor_timestamp` stuck at 1).
    """

    def __init__(self, period, pid):
        self.period = period
        self.pid = pid
        self.session = None
        self.sock = None
        self.parser = None
        self.window_end = 0.0
        self.collect_until = None
        self.samples = []
        self.next_start = boottime()

    @property
    def continuous(self):
        return self.period == 0

    def deadline(self):
        if self.session is None:
            return self.next_start
        if self.continuous:
            return float("inf")
        return self.collect_until if self.collect_until is not None else self.window_end

    def sockets(self):
        return [self.sock] if self.sock is not None else []

    def step(self, output, now):
        if self.session is None:
            if now >= self.next_start:
                self.start(now)
            return
        if self.continuous:
            return
        if now >= self.window_end or (self.collect_until is not None and now >= self.collect_until):
            self.finish(output)

    def start(self, now):
        # Continuous mode reaches next_start again only after a failure.
        self.next_start = now + (self.period or 30.0)
        self.session = request_session(HEART_RATE, self.pid)
        self.sock = connect_data_socket(self.session)
        self.parser = FrameParser(HEART_RATE_RECORD)
        self.samples = []
        self.collect_until = None
        self.window_end = now + HEART_RATE_WINDOW
        dbus(f"{MANAGER_PATH}/{HEART_RATE}", "local.HrmSensor.start", f"int32:{self.session}")

    def receive(self, output):
        assert self.sock is not None and self.parser is not None
        data = self.sock.recv(65536)
        if not data:
            raise RuntimeError("sensorfwd closed the heart-rate socket")
        records = self.parser.feed(data)
        if self.continuous:
            wall_now, boot_now = time.time(), boottime()
            for timestamp_us, status, bpm in records:
                reading = {
                    "sensor_timestamp": timestamp_us,
                    # 0 while the PPG warms up or the watch is off the wrist.
                    "bpm": bpm if bpm > 0 else None,
                    "status": status,
                    "samples": 1,
                }
                write_event(
                    output,
                    "heart_rate",
                    {"heart_rate": reading},
                    observed_epoch=sample_wall_time(timestamp_us, wall_now, boot_now),
                )
            output.flush()
            return
        for timestamp_us, status, bpm in records:
            # bpm reads 0 until the PPG sensor has warmed up (~11 s).
            if bpm > 0:
                self.samples.append((timestamp_us, status, bpm))
                if self.collect_until is None:
                    self.collect_until = boottime() + HEART_RATE_COLLECT
        self.step(output, boottime())

    def finish(self, output):
        if self.samples:
            bpms = sorted(sample[2] for sample in self.samples)
            timestamp_us, status, _ = self.samples[-1]
            reading = {
                "sensor_timestamp": timestamp_us,
                "bpm": bpms[len(bpms) // 2],
                "status": status,
                "samples": len(bpms),
            }
            observed = sample_wall_time(timestamp_us, time.time(), boottime())
        else:
            reading = {"sensor_timestamp": None, "bpm": None, "status": None, "samples": 0}
            observed = None
        write_event(output, "heart_rate", {"heart_rate": reading}, observed_epoch=observed)
        output.flush()
        self.release()

    def release(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        stop_and_release(HEART_RATE, "HrmSensor", self.session, self.pid)
        self.session = None
        self.collect_until = None


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, interrupt)
    if len(sys.argv) not in (3, 4, 5):
        usage()
        return 2

    output_path = sys.argv[1]
    try:
        duration = float(sys.argv[2])
        movement_interval = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
        heart_rate_period = float(sys.argv[4]) if len(sys.argv) == 5 else 0.0
    except ValueError:
        usage()
        return 2
    if duration <= 0 or movement_interval <= 0 or heart_rate_period < 0:
        usage()
        return 2

    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    pid = os.getpid()
    accelerometer_session = None
    heart_rate = HeartRateCycle(heart_rate_period, pid)
    accelerometer_path = f"{MANAGER_PATH}/{ACCELEROMETER}"

    try:
        with open(output_path, "a", encoding="utf-8") as output:
            accelerometer_session = request_session(ACCELEROMETER, pid)
            sock = connect_data_socket(accelerometer_session)
            dbus(
                accelerometer_path,
                "local.AccelerometerSensor.setInterval",
                f"int32:{accelerometer_session}",
                f"int32:{int(movement_interval * 1000)}",
            )
            dbus(accelerometer_path, "local.AccelerometerSensor.start", f"int32:{accelerometer_session}")
            parser = FrameParser()
            deadline = boottime() + duration
            next_battery_check = boottime()

            while True:
                now = boottime()
                if now >= deadline:
                    break
                if now >= next_battery_check:
                    capacity = battery_capacity()
                    next_battery_check = now + BATTERY_CHECK_EVERY
                    if capacity is not None and capacity < BATTERY_FLOOR:
                        write_event(output, "stopped", {"reason": "battery_floor", "capacity": capacity})
                        print(f"Stopped: battery at {capacity}%.", file=sys.stderr)
                        break
                try:
                    heart_rate.step(output, now)
                except RuntimeError as error:
                    # One failed reading must not end the night's movement record.
                    print(f"heart rate: {error}", file=sys.stderr)
                    heart_rate.release()

                wait = min(SELECT_TIMEOUT, deadline - now, heart_rate.deadline() - now)
                readable, _, _ = select.select([sock, *heart_rate.sockets()], [], [], max(0.0, wait))
                if heart_rate.sock is not None and heart_rate.sock in readable:
                    try:
                        heart_rate.receive(output)
                    except RuntimeError as error:
                        print(f"heart rate: {error}", file=sys.stderr)
                        heart_rate.release()
                if sock not in readable:
                    continue
                data = sock.recv(65536)
                if not data:
                    raise RuntimeError("sensorfwd closed the data socket")
                wall_now, boot_now = time.time(), boottime()
                for timestamp_us, x, y, z in parser.feed(data):
                    write_event(
                        output,
                        "movement",
                        {
                            "accelerometer": {
                                "sensor_timestamp": timestamp_us,
                                "x": round(x, 3),
                                "y": round(y, 3),
                                "z": round(z, 3),
                            }
                        },
                        observed_epoch=sample_wall_time(timestamp_us, wall_now, boot_now),
                    )
                output.flush()
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
    finally:
        heart_rate.release()
        stop_and_release(ACCELEROMETER, "AccelerometerSensor", accelerometer_session, pid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
