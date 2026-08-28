import json
import numpy as np
from datetime import datetime, timezone

PATH = "/home/martin/workspace/prototypes/sleep-analysis/sleep-20260827T201012Z.jsonl"
EPOCH_S = 30  # standard actigraphy epoch length

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
            hr_t.append(t)
            hr_bpm.append(d["heart_rate"]["bpm"])

mov_t = np.array(mov_t); mov_mag = np.array(mov_mag)
hr_t = np.array(hr_t); hr_bpm = np.array(hr_bpm)

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
    activity[i] = mov_diff[m].sum() if c else 0.0

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
log_act = np.log1p(activity)
from sklearn.cluster import KMeans
km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(log_act.reshape(-1,1))
centers = km2.cluster_centers_.flatten()
sleep_label = np.argmin(centers)  # lower activity cluster = sleep
is_sleep_raw = km2.labels_ == sleep_label

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

def rolling_std(a, win):
    half = win // 2
    out = np.empty_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i-half), min(len(a), i+half+1)
        out[i] = np.std(a[lo:hi])
    return out

hr_baseline = rolling_median(hr_epoch, win=61)   # ~30 min
hr_detrend = hr_epoch - hr_baseline
hr_var = rolling_std(hr_epoch, win=5)            # ~2.5 min

feat_idx = np.flatnonzero(sleep_mask)
X = np.column_stack([
    log_act[feat_idx],
    hr_detrend[feat_idx],
    hr_var[feat_idx],
])
# standardize
Xs = (X - X.mean(axis=0)) / X.std(axis=0)

km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs)
labels3 = km3.labels_
centers3 = km3.cluster_centers_  # in standardized space
# map back to real units for interpretation
centers3_real = centers3 * X.std(axis=0) + X.mean(axis=0)

# deep sleep: lowest activity + most-negative HR-vs-local-baseline
# REM: low activity + HR at/above local baseline and/or more beat-to-beat variability
# light: intermediate activity, the remaining cluster
order_by_hrdetrend = np.argsort(centers3_real[:,1])  # ascending detrended HR
deep_c = order_by_hrdetrend[0]
rem_c = order_by_hrdetrend[-1]
light_c = order_by_hrdetrend[1]

stage_names = {deep_c: "deep", rem_c: "rem", light_c: "light"}
epoch_stage = np.array(["wake"] * n_epochs, dtype=object)
for j, i in enumerate(feat_idx):
    epoch_stage[i] = stage_names[labels3[j]]

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
for name in ["light","deep","rem"]:
    n = (epoch_stage[window]==name).sum()
    pct = n/sleep_epochs_in_window*100 if sleep_epochs_in_window else 0
    print(f"  {name:5s}: {fmt_dur(n):8s} ({pct:.1f}% of sleep)")
print()
print("Cluster centers (log-activity, HR-vs-local-baseline, HR variability), by stage:")
for c,name in stage_names.items():
    print(f"  {name:5s}: activity(log)={centers3_real[c,0]:.2f}  hr_detrend={centers3_real[c,1]:+.1f} bpm  hr_var={centers3_real[c,2]:.2f}")

# hypnogram string, one char per epoch, 60 epochs (30 min) per line
sym = {"wake":"W","light":"L","deep":"D","rem":"R"}
line = "".join(sym[s] for s in epoch_stage)
print()
print("Hypnogram (30s/char, 30min/line):")
for i in range(0, len(line), 60):
    print(f"{ts(i):>6}  {line[i:i+60]}")

# ---- Arousal detection: joint movement + heart-rate change ----
# A single-epoch movement spike with no heart-rate response is just repositioning.
# A sustained *joint* rise in movement level and HR variability above the sleeping
# baseline is the signature of an actual arousal (stirring toward wake).
WIN = 10  # ~5 min of 30s epochs

def rolling_mean(a, win):
    kernel = np.ones(win) / win
    return np.convolve(a, kernel, mode="same")

def rolling_std_centered(a, win):
    half = win // 2
    out = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        lo, hi = max(0, i - half), min(len(a), i + half + 1)
        out[i] = np.std(a[lo:hi])
    return out

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

move_level = rolling_mean(activity, WIN)
hr_var_level = rolling_std_centered(hr_epoch, WIN)

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
