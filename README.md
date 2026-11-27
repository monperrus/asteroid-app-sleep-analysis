# AsteroidOS sleep recorder and analysis

Privacy-preserving sleep/wake research prototype for AsteroidOS. It combines
the watch-side recorder in [`app-sleep/`](app-sleep/) with a local analyzer.
It is not a medical device and does not provide clinical sleep staging.

## Recorder

[`app-sleep/`](app-sleep/) contains the launcher, system service, and sensor
collector installed on the watch. Its
[README](app-sleep/README.md) documents installation and recording.

Use `app-sleep/fetch-sleep-recording.sh` to retrieve the newest recording into
this repository, then analyze it with the command below. Recordings and
generated charts are private local artifacts and are not committed.

## Analysis

The analyzer estimates sleep patterns from movement and heart-rate data.

Movement is the primary signal; heart-rate trends and HRV should
only help distinguish quiet wakefulness from sleep. 

Background: See the AASM position statement
(https://aasm.org/consumer-sleep-technology-position-statement/) and
https://pubmed.ncbi.nlm.nih.gov/37338335/.

## Requirements

`pip install numpy scikit-learn plotly`

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

`--calibration PATH` reads or writes another calibration file instead of the
one next to the script.

Sleep onset is the first 10 min of immobility (no epoch with a movement
burst above 3x the night's median epoch peak), because lying awake in bed
stays under the wake/sleep activity threshold. The report warns when `deep`
is outside 10-35% of sleep; so far it is 53-76% on every recorded night,
because movement and HR cannot separate N2 from N3 (see `TODO.md`).

`--analyze` prints a text report (sleep onset/offset, time-in-stage,
hypnogram, arousal/spike listings) and writes `<stem>_stages.html`: 5 stacked
interactive time-series panels (raw movement, raw HR, and the three
clustering inputs — movement level, HR-detrended, HR variability) with the
classified stage shaded behind each one. Hover over a shaded stage region to
see its precise start/end and the characteristics (move_level, hr_detrend,
hr_var) that drove its classification. Open the file in a browser.

`calibration.json` and the generated `.html`/`.jsonl` files are local,
regeneratable/private artifacts and are gitignored — not checked in.

## Comparing with the Zeblaze watch

Wearing the Zeblaze on the same night gives a second staging (`zeblaze-ble
sleep`, synced daily into `sleep-YYYYMMDD-zeblaze.json`). Its own algorithm
is a consumer one, not polysomnography, and it never reports wake.

```
python3 analyze_sleep.py --analyze night.jsonl --epochs night_epochs.csv
python3 compare_reference.py night_epochs.csv sleep-YYYYMMDD-zeblaze.json [--json]
```

It prints the onset difference, each side's deep share and Cohen's kappa of
deep vs not-deep over the epochs both call sleep. It exits 2 when no
Zeblaze night overlaps the recording.

### Dual-wear night protocol

For the series pre-registered in `prereg/2026-11-27-deep-vs-zeblaze.md`
(5 nights).

Evening:
1. Charge the AsteroidOS watch to ≥ 95 %: the recorder stops at 9 %, which
   took ~7.2 h on the 2026-11 nights.
2. With the Zeblaze in BLE range of the laptop, enable its REM staging once
   (it reports no REM otherwise):
   `zeblaze-ble rem-set --i-understand-this-writes D6:45:15:30:04:71`
3. Wear one watch on each wrist; start the recorder from the "Start Sleep
   Recording (10h)" launcher entry.

Morning:
1. Fetch the recording: `app-sleep/fetch-sleep-recording.sh` over SSH, or
   over USB: `adb pull /home/ceres/sleep-recordings/sleep-<start>.jsonl .`
2. Sync the Zeblaze with the laptop in BLE range:
   `zeblaze-ble sleep D6:45:15:30:04:71 --i-understand-this-writes > sleep-$(date +%Y%m%d)-zeblaze.json`
   The daily anacron job (`zeblaze-sleep-sync.sh`) does the same, but only
   when the watch is in range; out of range it leaves an empty file. A file
   holds the last few nights, named by the sync day, not the night.
3. Score the night:
   ```
   python3 analyze_sleep.py --analyze sleep-<start>.jsonl --epochs sleep-<start>.csv
   python3 compare_reference.py sleep-<start>.csv sleep-<day>-zeblaze.json
   ```
   Keep the night if the overlap is ≥ 4 h; otherwise log it as excluded and
   record another one.

## Stages

`wake` / `light` / `deep` / `rem`, plus `unknown1`: movement-suppressed
epochs with HR variability too high to trust as deep sleep, but not high
enough to call REM. See `TODO.md` for why this is a separate label instead
of being folded into `deep` or `rem`, and the competing hypotheses for what
it actually is (NREM microarousals, apnea/hypopnea-linked HR cycling, or
under-detected REM).
