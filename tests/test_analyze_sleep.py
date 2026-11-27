"""End-to-end checks of analyze_sleep.py on synthetic nights."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "analyze_sleep.py"
START = datetime(2026, 11, 26, 21, 0, tzinfo=timezone.utc)

# Centers fitted with --calibrate on real nights (2026-08/11); fixed here so the
# test does not depend on the private local calibration.json.
CALIBRATION = {
    "wake_sleep": [[1.4420594459903997], [3.8383911528855883]],
    "quiet_active": [[1.4708444295670695], [2.950464796824408], [2.006249938863604]],
    "hr_mean": [0.22097053726169844, 1.0359877959780763],
    "hr_std": [2.4745065003502775, 1.1172530673787167],
    "deep_rem": [[-0.09326080121335649, -0.1184600176945907],
                 [3.2699568425433143, 4.153504370416684]],
}


def write_night(path: Path, restless_min: int, sleep_min: int = 180, seed: int = 0) -> None:
    """1 Hz accelerometer, 0.5 Hz HR. The first `restless_min` minutes are
    quiet wake in bed: resting-level jitter (mean activity stays under the
    wake threshold) plus one 300-unit jerk every 3 minutes."""
    rng = np.random.default_rng(seed)
    total_s = (restless_min + sleep_min) * 60
    with path.open("w") as f:
        for s in range(total_s):
            t = START + timedelta(seconds=s)
            z = 1000 + rng.normal(0, 2.5)
            if s < restless_min * 60 and s % 180 == 90:
                z += 300
            f.write(json.dumps({"type": "movement",
                                "observed_at": t.isoformat().replace("+00:00", "Z"),
                                "accelerometer": {"x": 0.0, "y": 0.0, "z": z}}) + "\n")
            if s % 2 == 0:
                f.write(json.dumps({"type": "heart_rate",
                                    "observed_at": t.isoformat().replace("+00:00", "Z"),
                                    "heart_rate": {"bpm": round(56 + rng.normal(0, 0.5))}}) + "\n")


def analyze(tmp_path: Path, restless_min: int) -> str:
    night = tmp_path / "night.jsonl"
    write_night(night, restless_min)
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps(CALIBRATION))
    out = subprocess.run([sys.executable, str(SCRIPT), "--analyze", str(night), "--calibration", str(calib)],
                         capture_output=True, text=True, check=True)
    return out.stdout


def onset_epoch(report: str) -> int:
    m = re.search(r"Sleep onset\s*: epoch (\d+)", report)
    assert m, report
    return int(m.group(1))


def test_quiet_wake_with_movement_bursts_is_not_sleep(tmp_path: Path) -> None:
    # last burst at 19.5 min, so 10 min of immobility first ends at 29.5 min
    onset = onset_epoch(analyze(tmp_path, restless_min=20))
    assert 39 <= onset <= 41  # 19.5-20.5 min, in 30 s epochs


def test_still_from_the_start_is_asleep_from_the_start(tmp_path: Path) -> None:
    assert onset_epoch(analyze(tmp_path, restless_min=0)) == 0


def test_implausible_deep_share_is_flagged(tmp_path: Path) -> None:
    # perfectly still with a flat HR: the classifier calls nearly all of it deep
    report = analyze(tmp_path, restless_min=0)
    assert re.search(r"WARNING: deep \d+% is outside the plausible 10-35%", report), report
