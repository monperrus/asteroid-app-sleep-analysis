# Pre-registration: deep sleep (N3) of analyze_sleep.py vs a Zeblaze dual-wear reference

Registered 2026-11-27, before any dual-wear night of this series is recorded.

## Question
Can a feature fixed in advance make `analyze_sleep.py`'s `deep` agree with the
Zeblaze watch's `deep` well enough to report it, or must `deep` be reported
as "deep or light"?

## Data
- Nights: the next 5 nights with both watches on (AsteroidOS recorder on one
  wrist, Zeblaze on the other), REM enabled on the Zeblaze (`zeblaze-ble
  rem-set`), recorder started at ≥ 95 % battery. Nights where either side has
  < 4 h of overlap are excluded and replaced; the exclusion is logged.
- Already in hand, not part of the 5: 2026-09-01 (dev night, used to build
  the comparison tool and to pick the candidates below).
- Scoring: `compare_reference.py` at commit 4a2559e, analyzer calibration
  `calibration.json` as of 2026-11-27, unchanged during the series.

## Statistic
Per night: Cohen's kappa of deep vs not-deep over epochs both call sleep
(`deep_kappa`), and `|deep_pct_ours - deep_pct_zeblaze|` in points.

## Predictions (current analyzer, before any change)
- P1: `deep_kappa` < 0.20 on every night (dev night: 0.10).
- P2: `deep_pct_ours` ≥ 45 % on every night; `deep_pct_zeblaze` 15-35 %.
- P3: onset differs by ≤ 15 min on ≥ 4 of 5 nights (dev night: +6 min).

## Candidates (fixed now, no others)
A still-tier epoch is `deep` only if, in addition to the current rule:
- C1: its HR is below the 25th percentile of the night's still-tier HR.
- C2: C1, and it lies in the first half of the sleep period.
- C3: C1, and it belongs to a run of ≥ 20 such epochs (10 min).

## Decision rule
Leave-one-night-out: for each held-out night, pick the candidate with the
best mean `deep_kappa` on the other 4, and score it on the held-out night.
- ADOPT the candidate chosen most often if the held-out `deep_kappa` mean
  is ≥ 0.40 and ≥ the current analyzer's + 0.20, and the median deep-share
  error is ≤ 10 points.
- Otherwise STOP: report `deep` and `light` merged as "N2/N3" and drop the
  deep share from the report.
