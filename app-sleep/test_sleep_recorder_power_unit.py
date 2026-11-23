"""Contract checks for the recorder's temporary sensor power dependency."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parent


def test_recorder_scopes_sensorfwd_to_its_lifetime() -> None:
    recorder = (ROOT / "sleep-recorder.service").read_text()
    assert "ExecStartPre=+/bin/systemctl start sensorfwd.service" in recorder
    assert "ExecStopPost=+/bin/systemctl stop sensorfwd.service" in recorder


def test_recorder_has_no_sensorfwd_requirement_that_can_deadlock_shutdown() -> None:
    recorder = (ROOT / "sleep-recorder.service").read_text()
    assert "Requires=sensorfwd.service" not in recorder
    assert "sleep-recorder-power.service" not in recorder
