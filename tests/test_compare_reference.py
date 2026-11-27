"""Validate compare_reference.py on nights whose answer is known."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compare_reference as cr

T0 = 1_788_215_760  # a minute boundary
# 30 min light, 20 min deep, 10 min light
NIGHT = {"date": "2026-08-31", "data": {
    "start_timestamp": T0, "end_timestamp": T0 + 3600,
    "stages": [{"start_timestamp": T0, "duration_minutes": 30, "stage": 1},
               {"start_timestamp": T0 + 1800, "duration_minutes": 20, "stage": 2},
               {"start_timestamp": T0 + 3000, "duration_minutes": 10, "stage": 1},
               {"start_timestamp": T0 + 3600, "duration_minutes": 0, "stage": 0}]}}


def epochs_from(stage_of_minute) -> list[tuple[float, str]]:
    return [(T0 + 30 * i, stage_of_minute(i // 2)) for i in range(120)]


def ref(m: int) -> str:
    return "deep" if 30 <= m < 50 else "light"


def test_identical_staging_agrees_perfectly() -> None:
    s = cr.score(epochs_from(ref), NIGHT)
    assert s["deep_kappa"] == 1.0
    assert s["deep_pct_ours"] == s["deep_pct_zeblaze"] == 100 * 20 / 60
    assert s["onset_diff_min"] == 0
    assert s["scored_epochs"] == 120


def test_inverted_deep_disagrees_completely() -> None:
    # no epoch agrees; kappa is -1 only with equal marginals, here 40 vs 20 deep
    # epochs: po=0, pe=2*(2/3)*(1/3)=4/9, kappa=-(4/9)/(5/9)
    s = cr.score(epochs_from(lambda m: "light" if ref(m) == "deep" else "deep"), NIGHT)
    assert s["deep_kappa"] == pytest.approx(-0.8)


def test_wake_before_onset_shifts_onset_and_is_not_scored() -> None:
    s = cr.score(epochs_from(lambda m: "wake" if m < 10 else ref(m)), NIGHT)
    assert s["onset_diff_min"] == 10
    assert s["scored_epochs"] == 100


def test_epoch_starting_on_a_stage_boundary_takes_the_new_stage() -> None:
    # epoch starting at minute 30 has its midpoint in the deep segment
    s = cr.score([(T0 + 1800, "deep")], NIGHT)
    assert s["deep_kappa"] is None  # one class only: kappa undefined, not 1
    assert s["deep_pct_zeblaze"] == 100


def test_no_overlapping_night_exits_2(tmp_path: Path) -> None:
    csv = tmp_path / "e.csv"
    csv.write_text(f"start_unix,stage\n{T0 + 86400},deep\n")
    z = tmp_path / "z.json"
    z.write_text(json.dumps([NIGHT]))
    assert cr.main([str(csv), str(z)]) == 2
