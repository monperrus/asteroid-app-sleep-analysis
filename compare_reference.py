"""Score analyze_sleep.py stages against a Zeblaze watch's staging of the same night.

  analyze_sleep.py --analyze night.jsonl --epochs night_epochs.csv
  compare_reference.py night_epochs.csv sleep-YYYYMMDD-zeblaze.json [--json]

The Zeblaze night that overlaps the recording most is used. Each 30 s epoch
takes the Zeblaze stage at its midpoint. The Zeblaze is a consumer watch with
its own movement+PPG algorithm, not polysomnography: agreement with it is a
second opinion, not accuracy. It never reports wake, and reports REM only
once `zeblaze-ble rem-set` has enabled it.

Exit status: 0 scored, 2 no Zeblaze night overlaps the recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ZEBLAZE_STAGE = {1: "light", 2: "deep", 3: "rem"}  # 0 marks the end of the night


def load_epochs(path: Path) -> list[tuple[float, str]]:
    with path.open() as f:
        return [(float(r["start_unix"]), r["stage"]) for r in csv.DictReader(f)]


def zeblaze_segments(night: dict) -> list[tuple[float, float, str]]:
    return [(s["start_timestamp"], s["start_timestamp"] + 60 * s["duration_minutes"], ZEBLAZE_STAGE[s["stage"]])
            for s in night["data"]["stages"] if s["stage"] in ZEBLAZE_STAGE]


def pick_night(nights: list[dict], t0: float, t1: float) -> dict | None:
    def overlap(n: dict) -> float:
        return min(t1, n["data"]["end_timestamp"]) - max(t0, n["data"]["start_timestamp"])
    best = max(nights, key=overlap, default=None)
    return best if best is not None and overlap(best) > 0 else None


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa of two binary labelings; None when chance agreement is total."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(epochs: list[tuple[float, str]], night: dict, epoch_s: float = 30.0) -> dict:
    segs = zeblaze_segments(night)
    ref_start, ref_end = night["data"]["start_timestamp"], night["data"]["end_timestamp"]

    def ref_stage(t: float) -> str | None:
        for a, b, st in segs:
            if a <= t < b:
                return st
        return None

    pairs = [(ours, ref_stage(t + epoch_s / 2)) for t, ours in epochs]
    both = [(o, r) for o, r in pairs if r is not None and o != "wake"]
    ours_onset = next((t for t, st in epochs if st != "wake"), None)
    rec_end = epochs[-1][0] + epoch_s
    return {
        "zeblaze_date": night["date"],
        "zeblaze_rem_enabled": any(st == "rem" for *_, st in segs),
        "onset_ours": ours_onset,
        "onset_zeblaze": ref_start,
        "onset_diff_min": None if ours_onset is None else (ours_onset - ref_start) / 60,
        "overlap_min": (min(rec_end, ref_end) - max(epochs[0][0], ref_start)) / 60,
        "scored_epochs": len(both),
        "deep_pct_ours": 100 * sum(o == "deep" for o, _ in both) / len(both) if both else None,
        "deep_pct_zeblaze": 100 * sum(r == "deep" for _, r in both) / len(both) if both else None,
        "deep_kappa": kappa([o == "deep" for o, _ in both], [r == "deep" for _, r in both]),
    }


def fmt_time(t: float | None) -> str:
    return "-" if t is None else datetime.fromtimestamp(t).astimezone().strftime("%H:%M")


def fmt_num(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("epochs", type=Path, help="CSV written by analyze_sleep.py --epochs")
    p.add_argument("zeblaze", type=Path, help="JSON written by zeblaze-ble sleep")
    p.add_argument("--json", action="store_true", help="print the scores as one JSON object")
    a = p.parse_args(argv)

    epochs = load_epochs(a.epochs)
    night = pick_night(json.loads(a.zeblaze.read_text()), epochs[0][0], epochs[-1][0] + 30)
    if night is None:
        print(f"{a.zeblaze}: no night overlaps the recording", file=sys.stderr)
        return 2
    s = score(epochs, night)
    if a.json:
        print(json.dumps(s))
        return 0
    print(f"Zeblaze night {s['zeblaze_date']} (REM {'on' if s['zeblaze_rem_enabled'] else 'off'}), "
          f"overlap {s['overlap_min']:.0f} min, {s['scored_epochs']} epochs scored")
    print(f"onset  ours {fmt_time(s['onset_ours'])}  zeblaze {fmt_time(s['onset_zeblaze'])}  "
          f"diff {fmt_num(s['onset_diff_min'], '+.0f')} min")
    print(f"deep   ours {fmt_num(s['deep_pct_ours'], '.0f')}%  zeblaze {fmt_num(s['deep_pct_zeblaze'], '.0f')}%  "
          f"kappa {fmt_num(s['deep_kappa'], '.2f')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
