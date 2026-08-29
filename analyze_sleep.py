"""Actigraphy + heart-rate sleep-stage classifier.

Two-step workflow, see README.md for details:
  analyze_sleep.py --calibrate <file.jsonl>   fit thresholds, save calibration.json
  analyze_sleep.py --analyze   <file.jsonl>   classify against saved thresholds,
                                               print a report, write <stem>_stages.png
"""

import argparse
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = "/home/martin/workspace/prototypes/sleep-analysis/sleep-20260827T201012Z.jsonl"
CALIBRATION_PATH = Path(__file__).with_name("calibration.json")
EPOCH_S = 30  # standard actigraphy epoch length

def parse_args():
    p = argparse.ArgumentParser(description="Classify a night's movement+HR recording into sleep stages.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", action="store_true",
                       help="fit stage-classification thresholds on PATH and save them to calibration.json")
    mode.add_argument("--analyze", action="store_true",
                       help="classify PATH against thresholds saved by --calibrate, instead of refitting per run")
    p.add_argument("path", nargs="?", default=DEFAULT_PATH, help="sleep data jsonl file")
    return p.parse_args()

args = parse_args()
PATH = args.path
calib = {} if args.calibrate else json.loads(CALIBRATION_PATH.read_text())

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

mov_t, mov_mag = [], []
hr_t, hr_bpm = [], []

with open(PATH) as f:
    for line in f:
        d = json.loads(line)
        t = parse_ts(d["observed_at"])
        if d["type"] == "movement":
            a = d["accelerometer"]
            mag = (a["x"]**2 + a["y"]**2 + a["z"]**2) ** 0.5
            mov_t.append(t)
            mov_mag.append(mag)
        elif d["type"] == "heart_rate":
            bpm = d["heart_rate"]["bpm"]
            hr_t.append(t)
            hr_bpm.append(bpm if bpm is not None else np.nan)

mov_t = np.array(mov_t); mov_mag = np.array(mov_mag)
hr_t = np.array(hr_t); hr_bpm = np.array(hr_bpm, dtype=float)

t0, t1 = mov_t.min(), mov_t.max()
n_epochs = int(np.ceil((t1 - t0) / EPOCH_S))
edges = t0 + np.arange(n_epochs + 1) * EPOCH_S

# jerk-based activity metric: sum of abs successive differences in magnitude per epoch
mov_diff = np.abs(np.diff(mov_mag))
mov_diff_t = mov_t[1:]

activity = np.zeros(n_epochs)
counts = np.zeros(n_epochs)
for i in range(n_epochs):
    m = (mov_diff_t >= edges[i]) & (mov_diff_t < edges[i+1])
    c = m.sum()
    counts[i] = c
    # mean, not sum: sample spacing is irregular (~1-7s gaps), so summing raw
    # diffs would confound sample density with true movement level.
    activity[i] = mov_diff[m].mean() if c else 0.0

# hr per epoch: nearest-sample forward/backward fill
hr_epoch = np.full(n_epochs, np.nan)
for i in range(n_epochs):
    mid = (edges[i] + edges[i+1]) / 2
    idx = np.searchsorted(hr_t, mid)
    cands = []
    if idx < len(hr_t): cands.append(idx)
    if idx > 0: cands.append(idx-1)
    if cands:
        best = min(cands, key=lambda j: abs(hr_t[j]-mid))
        hr_epoch[i] = hr_bpm[best]
# interpolate remaining NaNs
nan_mask = np.isnan(hr_epoch)
if nan_mask.any():
    valid = ~nan_mask
    hr_epoch[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(valid), hr_epoch[valid])

# ---- Wake/Sleep classification ----
from sklearn.cluster import KMeans

def cluster(X, name, n_clusters=2):
    """n-way split of X: fit a fresh k-means and record its centers in
    --calibrate mode, or assign to the nearest previously-saved center in
    --analyze mode, so stage thresholds are fixed across nights instead of
    being refit (and potentially drifting) on every run."""
    if args.calibrate:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
        centers = km.cluster_centers_
        calib[name] = centers.tolist()
        return km.labels_, centers
    centers = np.array(calib[name])
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1), centers

log_act = np.log1p(activity)
labels2, centers2 = cluster(log_act.reshape(-1, 1), "wake_sleep")
sleep_label = np.argmin(centers2.flatten())  # lower activity cluster = sleep
is_sleep_raw = labels2 == sleep_label

# smooth: majority vote over 3-epoch window to remove single-epoch flicker
def smooth_binary(arr, win=3):
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i-win//2), min(len(arr), i+win//2+1)
        out[i] = np.round(arr[lo:hi].mean())
    return out.astype(bool)

is_sleep = smooth_binary(is_sleep_raw.astype(float), win=3)

# ---- Determine sleep period (onset to offset), allowing WASO in between ----
sleep_idx = np.flatnonzero(is_sleep)
if len(sleep_idx) == 0:
    raise SystemExit("No sleep detected")
onset_i, offset_i = sleep_idx[0], sleep_idx[-1]

# require sustained sleep for onset (>=5 consecutive min = 10 epochs) to avoid false start
def find_onset(is_sleep, min_epochs=10):
    run = 0
    for i, v in enumerate(is_sleep):
        if v:
            run += 1
            if run >= min_epochs:
                return i - min_epochs + 1
        else:
            run = 0
    return int(np.flatnonzero(is_sleep)[0])

onset_i = find_onset(is_sleep)
offset_i = len(is_sleep) - 1 - find_onset(is_sleep[::-1].copy())

window = slice(onset_i, offset_i + 1)
in_bed_epochs = offset_i - onset_i + 1
sleep_epochs_in_window = is_sleep[window].sum()
waso_epochs = in_bed_epochs - sleep_epochs_in_window

# ---- Stage classification within sleep epochs only ----
sleep_mask = is_sleep.copy()
sleep_mask[:onset_i] = False
sleep_mask[offset_i+1:] = False

# HR drifts down slowly over the night with core body temperature (circadian),
# which would otherwise dominate a raw-HR clustering and just split early-vs-late
# night instead of stage-related patterns. Detrend against a ~30min rolling
# median, and add short-term HR variability (beat-to-beat bpm dispersion is a
# crude proxy for the HRV signal we don't have) since REM is associated with
# transient HR elevation/variability relative to surrounding NREM.
def rolling_median(a, win):
    half = win // 2
    out = np.empty_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i-half), min(len(a), i+half+1)
        out[i] = np.median(a[lo:hi])
    return out

def rolling_mean(a, win):
    return np.convolve(a, np.ones(win) / win, mode="same")

def rolling_std_centered(a, win):
    half = win // 2
    out = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        lo, hi = max(0, i - half), min(len(a), i + half + 1)
        out[i] = np.std(a[lo:hi])
    return out

hr_baseline = rolling_median(hr_epoch, win=61)      # ~30 min
hr_detrend = hr_epoch - hr_baseline
hr_var = rolling_std_centered(hr_epoch, win=5)      # ~2.5 min, beat-to-beat proxy
move_level = rolling_mean(activity, win=10)         # ~5 min movement level

# REM is, by definition, near-total skeletal-muscle atonia (movement suppressed)
# *combined with* an autonomically active, variable heart rate. A rise in HR
# variability alone is NOT REM if movement is also elevated - that combination is
# the actigraphy signature of an arousal/awakening instead (motor tone returning).
# So gate REM out of the high-movement epochs *before* clustering on HR, rather
# than letting an unconstrained clustering pick REM by HR alone.
#
# A plain 2-way split on move_level was too blunt in both directions:
#  - real REM atonia is regularly interrupted by brief phasic muscle twitches,
#    which lift the 5-minute move_level just enough to cross a 2-cluster
#    boundary, dumping those epochs into "light" with no chance to be scored
#    on HR - even when their HR trace was screaming REM.
#  - conversely, a movement bump with a *flat* HR response (no autonomic
#    activation) is real body movement, which by definition rules out N3
#    "deep" sleep too - it should read as "light", not get swallowed into
#    "deep" just because HR alone still looks calm.
# 3-way split on move_level separates: "still" (can be deep or REM by HR),
# "twitch" (movement-elevated but not sustained - can only be *promoted* to
# REM by a genuine HR signature, otherwise stays "light"), and "active"
# (sustained movement - always "light", never scored on HR at all).
sleep_idx2 = np.flatnonzero(sleep_mask)
move_labels, move_centers = cluster(
    np.log1p(move_level[sleep_idx2]).reshape(-1, 1), "quiet_active", n_clusters=3
)
low_label, mid_label, active_label = np.argsort(move_centers.flatten())

still_mask = np.zeros(n_epochs, dtype=bool)
still_mask[sleep_idx2[move_labels == low_label]] = True
twitch_mask = np.zeros(n_epochs, dtype=bool)
twitch_mask[sleep_idx2[move_labels == mid_label]] = True
active_mask = sleep_mask & ~still_mask & ~twitch_mask

epoch_stage = np.array(["wake"] * n_epochs, dtype=object)
epoch_stage[active_mask] = "light"
epoch_stage[twitch_mask] = "light"  # default; promoted to "rem" below if HR matches

# Among the still (movement-suppressed) epochs, deep vs REM is a pure HR
# question: deep = HR at/below local baseline and stable; REM = HR at/above
# baseline and/or more beat-to-beat variable. The same fitted deep/REM
# centers are then applied to the twitch tier below, but only to *promote*
# to REM - a twitch-tier epoch nearest the "deep" center just stays "light",
# since flat HR plus real movement was never a deep-sleep candidate.
still_idx = np.flatnonzero(still_mask)
Xq = np.column_stack([hr_detrend[still_idx], hr_var[still_idx]])
if args.calibrate:
    hr_mean, hr_std = Xq.mean(axis=0), Xq.std(axis=0)
    calib["hr_mean"], calib["hr_std"] = hr_mean.tolist(), hr_std.tolist()
else:
    hr_mean, hr_std = np.array(calib["hr_mean"]), np.array(calib["hr_std"])
Xqs = (Xq - hr_mean) / hr_std
hr_labels, hr_centers = cluster(Xqs, "deep_rem")
hr_centers_real = hr_centers * hr_std + hr_mean
deep_label = np.argmin(hr_centers_real[:, 0])  # more negative hr_detrend = deep
rem_label = 1 - deep_label

# "unknown1": movement-suppressed (still tier, so genuinely not REM-tier
# movement and not repositioning) but with HR variability well above the
# calibrated deep baseline, without crossing all the way into the REM
# cluster. Nearest-center deep/REM assignment would silently fold this into
# "deep", which is wrong on the classifier's own terms - deep/N3 is defined
# by autonomic *quiescence* as much as by stillness, and this regime isn't
# quiescent. We deliberately do NOT auto-resolve it to "rem" or "deep" and
# instead surface it as its own label, because actigraphy + HR alone cannot
# distinguish between (at least) two physiologically different explanations:
#   (a) NREM cyclic alternating pattern (CAP) / recurring microarousals - a
#       normal autonomic microstructure of light/deep NREM that modulates HR
#       cyclically over minutes without ever producing the sustained movement
#       response the joint movement+HR arousal detector requires; or
#   (b) cyclic variation of heart rate (CVHR) from repeated brief respiratory
#       events (apnea/hypopnea) - the classic surge-recover HR pattern seen
#       with sleep-disordered breathing, which likewise leaves no actigraphy
#       movement signature.
# Telling these apart needs a signal this recording doesn't have (SpO2 or
# respiration/airflow). See TODO.md for a third, competing hypothesis (that
# this is in fact under-detected REM) and why we're not folding it into
# "rem" outright despite that.
UNSTABLE_HR_VAR_MULT = 1.8  # multiple of the calibrated deep hr_var center
deep_hr_var_center = hr_centers_real[deep_label, 1]

for j, i in enumerate(still_idx):
    if hr_labels[j] == rem_label:
        epoch_stage[i] = "rem"
    elif hr_var[i] > UNSTABLE_HR_VAR_MULT * deep_hr_var_center:
        epoch_stage[i] = "unknown1"
    else:
        epoch_stage[i] = "deep"

twitch_idx = np.flatnonzero(twitch_mask)
Xt = np.column_stack([hr_detrend[twitch_idx], hr_var[twitch_idx]])
Xts = (Xt - hr_mean) / hr_std
d2 = ((Xts[:, None, :] - hr_centers[None, :, :]) ** 2).sum(axis=2)
twitch_labels = d2.argmin(axis=1)
for j, i in enumerate(twitch_idx):
    if twitch_labels[j] == rem_label:
        epoch_stage[i] = "rem"

# ---- Report ----
def fmt_dur(epochs):
    secs = epochs * EPOCH_S
    h, rem = divmod(secs, 3600)
    m, _ = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m"

def ts(i):
    return datetime.fromtimestamp(t0 + i*EPOCH_S, tz=timezone.utc).astimezone().strftime("%H:%M")

print(f"Recording span   : {datetime.fromtimestamp(t0,tz=timezone.utc).astimezone()} -> {datetime.fromtimestamp(t1,tz=timezone.utc).astimezone()}")
print(f"Recording length : {fmt_dur(n_epochs)}")
print(f"Sleep onset       : epoch {onset_i} (~{ts(onset_i)})")
print(f"Sleep offset      : epoch {offset_i} (~{ts(offset_i)}) [recording ends here, may not be final wake]")
print(f"Time in bed window: {fmt_dur(in_bed_epochs)}")
print(f"WASO (wake after onset): {fmt_dur(waso_epochs)}")
print(f"Total sleep time  : {fmt_dur(sleep_epochs_in_window)}")
print(f"Sleep efficiency  : {sleep_epochs_in_window/in_bed_epochs*100:.1f}%")
print()
for name in ["light","deep","rem","unknown1"]:
    n = (epoch_stage[window]==name).sum()
    pct = n/sleep_epochs_in_window*100 if sleep_epochs_in_window else 0
    print(f"  {name:8s}: {fmt_dur(n):8s} ({pct:.1f}% of sleep)")
print()
print("Stage cluster centers (move_level, hr_detrend, hr_var), by stage:")
for name in ["deep", "rem", "light", "unknown1"]:
    m = (epoch_stage[window] == name)
    if m.sum():
        print(f"  {name:8s}: move_level={move_level[window][m].mean():6.1f}  "
              f"hr_detrend={hr_detrend[window][m].mean():+.1f} bpm  hr_var={hr_var[window][m].mean():.2f}")

# hypnogram string, one char per epoch, 60 epochs (30 min) per line
sym = {"wake":"W","light":"L","deep":"D","rem":"R","unknown1":"U"}
line = "".join(sym[s] for s in epoch_stage)
print()
print("Hypnogram (30s/char, 30min/line):")
for i in range(0, len(line), 60):
    print(f"{ts(i):>6}  {line[i:i+60]}")

# ---- Arousal detection: joint movement + heart-rate change ----
# A single-epoch movement spike with no heart-rate response is just repositioning.
# A sustained *joint* rise in movement level and HR variability above the sleeping
# baseline is the signature of an actual arousal (stirring toward wake) - the
# mirror image of REM, which requires movement to stay LOW (see stage
# classification above).
WIN = 10  # ~5 min of 30s epochs

def find_runs(mask, min_len):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs

hr_var_level = rolling_std_centered(hr_epoch, WIN)  # coarser than hr_var (win=5), for sustained-rise detection

move_base = np.median(move_level[sleep_mask])
hr_var_base = np.median(hr_var_level[sleep_mask])

is_arousal = sleep_mask & (move_level > 1.5 * move_base) & (hr_var_level > 1.3 * hr_var_base)
arousal_runs = find_runs(is_arousal, min_len=6)  # sustained >= 3 min in both signals

print()
print(f"Sleeping baseline : activity~{move_base:.0f} (5min mean)  hr_std~{hr_var_base:.2f} bpm (5min window)")
print("Joint movement+HR arousal windows (sustained rise in both signals, likely stirring/waking):")
if not arousal_runs:
    print("  none detected")
for s, e in arousal_runs:
    dur = fmt_dur(e - s + 1)
    print(f"  {ts(s)}-{ts(e)}  ({dur})  activity~{move_level[s:e+1].mean():.0f} (x{move_level[s:e+1].mean()/move_base:.1f} baseline)"
          f"  hr_std~{hr_var_level[s:e+1].mean():.2f} (x{hr_var_level[s:e+1].mean()/hr_var_base:.1f} baseline)"
          f"  hr {hr_epoch[s:e+1].min():.0f}-{hr_epoch[s:e+1].max():.0f} bpm")

# movement-only spikes: brief high-activity epochs during sleep that never made it
# into an arousal run above (no sustained HR co-response) -> likely repositioning
spike_mask = sleep_mask & (activity > 5 * np.median(activity[sleep_mask])) & ~is_arousal
spike_idx = np.flatnonzero(spike_mask)
print()
print("Movement-only spikes (no sustained HR response - likely repositioning):")
if len(spike_idx) == 0:
    print("  none detected")
for i in spike_idx:
    print(f"  {ts(i)}  activity={activity[i]:.0f}  hr={hr_epoch[i]:.0f} bpm")

if args.calibrate:
    CALIBRATION_PATH.write_text(json.dumps(calib, indent=2))
    print(f"\nCalibration saved to {CALIBRATION_PATH}")

# ---- Figure: raw signals + clustering inputs, shaded by classified stage ----
if args.analyze:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    STAGE_COLOR = {"wake": "#9e9e9e", "light": "#4c9be8", "deep": "#1b3a6b", "rem": "#e8734c", "unknown1": "#f2c744"}

    def to_dt(ts_arr):
        return [datetime.fromtimestamp(x, tz=timezone.utc).astimezone() for x in ts_arr]

    epoch_centers = t0 + (np.arange(n_epochs) + 0.5) * EPOCH_S
    epoch_dt = to_dt(epoch_centers)
    mov_dt = to_dt(mov_t)
    hr_dt = to_dt(hr_t)

    # merge consecutive same-stage epochs into runs so shading is a handful of
    # axvspans, not one per epoch
    stage_runs, run_start = [], 0
    for i in range(1, n_epochs + 1):
        if i == n_epochs or epoch_stage[i] != epoch_stage[run_start]:
            stage_runs.append((run_start, i, epoch_stage[run_start]))
            run_start = i

    def shade_stages(ax):
        for s, e, stage in stage_runs:
            ax.axvspan(epoch_dt[s], epoch_dt[min(e, n_epochs - 1)], color=STAGE_COLOR[stage], alpha=0.3, lw=0)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(mov_dt, mov_mag, lw=0.3, color="black")
    axes[0].set_ylabel("accel mag")
    axes[0].set_title("Raw movement magnitude")

    axes[1].plot(hr_dt, hr_bpm, ".", ms=2, color="black")
    axes[1].set_ylabel("bpm")
    axes[1].set_title("Raw heart rate")

    axes[2].plot(epoch_dt, move_level, color="black", lw=0.8)
    axes[2].set_ylabel("move level")
    axes[2].set_title("Movement level (quiet/active clustering input)")

    axes[3].plot(epoch_dt, hr_detrend, color="black", lw=0.8)
    axes[3].set_ylabel("hr - baseline")
    axes[3].set_title("HR detrended (deep/REM clustering input)")

    axes[4].plot(epoch_dt, hr_var, color="black", lw=0.8)
    axes[4].set_ylabel("hr std")
    axes[4].set_title("HR variability (deep/REM clustering input)")

    for ax in axes:
        shade_stages(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=epoch_dt[0].tzinfo))
    axes[-1].set_xlabel("time")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in STAGE_COLOR.values()]
    fig.legend(legend_handles, STAGE_COLOR.keys(), loc="upper right", ncol=4, title="stage")
    fig.suptitle(f"Sleep stages — {Path(PATH).stem}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig_path = Path(PATH).parent / f"{Path(PATH).stem}_stages.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nFigure saved to {fig_path}")
