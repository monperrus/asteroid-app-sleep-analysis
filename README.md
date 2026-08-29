# Sleep analysis

We analyze sleep patterns

Movement is the primary signal; heart-rate trends and HRV should
only help distinguish quiet wakefulness from sleep. 

Background: See the AASM position statement
(https://aasm.org/consumer-sleep-technology-position-statement/) and
https://pubmed.ncbi.nlm.nih.gov/37338335/.

## Requirements

`pip install numpy scikit-learn matplotlib`

## Usage

Input is a `.jsonl` recording of `movement` (accelerometer) and `heart_rate`
events, one JSON object per line (see either `sleep-*.jsonl` file for the
shape).

Two-step workflow — thresholds are fit once, then reused, so classification
stays fixed across nights instead of silently refitting (and drifting) every
run:

```
python3 analyze_sleep.py --calibrate night.jsonl   # fits stage thresholds, writes calibration.json
python3 analyze_sleep.py --analyze   night.jsonl   # classifies against calibration.json
```

`--analyze` prints a text report (sleep onset/offset, time-in-stage,
hypnogram, arousal/spike listings) and writes `<stem>_stages.png`: 5 stacked
time-series panels (raw movement, raw HR, and the three clustering inputs —
movement level, HR-detrended, HR variability) with the classified stage
shaded behind each one.

`calibration.json` and the generated `.png`/`.jsonl` files are local,
regeneratable/private artifacts and are gitignored — not checked in.

## Stages

`wake` / `light` / `deep` / `rem`, plus `unknown1`: movement-suppressed
epochs with HR variability too high to trust as deep sleep, but not high
enough to call REM. See `TODO.md` for why this is a separate label instead
of being folded into `deep` or `rem`, and the competing hypotheses for what
it actually is (NREM microarousals, apnea/hypopnea-linked HR cycling, or
under-detected REM).
