# TODO

## REM bout duration calibration

`analyze_sleep.py` currently reports raw 30s-epoch REM runs. Sleep science
(Carskadon & Dement sleep-cycle literature) says real REM periods don't look
like this:

- First REM period of the night: ~10 min (rarely less than ~5).
- Later REM periods lengthen across the night: ~15-20 min in cycle 2, often
  20-40+ min in later cycles, with the longest REM bouts (30-60 min) in the
  last cycle before natural waking.
- Sub-2-minute "REM" isn't a recognized category — that short, it's almost
  certainly a transient arousal or HR-sensor noise, not REM.

Tonight's run produced 15 REM bouts, most 0.5-2 min, only two near 4.5 min —
all shorter than even a plausible first REM period. The 11.7%-of-TST
aggregate is probably more trustworthy than any individual reported bout.

**To do:** add a minimum-duration merge/filter to `epoch_stage` before
reporting REM bouts - e.g. merge REM epochs separated by <=1-2 min of
non-REM, then drop merged bouts still under ~5 min (or flag them separately
as "possible REM, sub-threshold duration" rather than reporting them as REM
outright). Re-validate the resulting bout-length distribution against the
ranges above before trusting per-bout output.

## "unknown1" stage: unresolved HR-instability regime

`analyze_sleep.py` labels an epoch `unknown1` when movement is suppressed
(same "still" tier eligible for deep/REM) but HR variability sits well above
the calibrated deep baseline (>1.8x `deep_hr_var_center`, see
`UNSTABLE_HR_VAR_MULT`) without crossing into the REM cluster. It is
deliberately never auto-resolved to `deep` or `rem`.

This was first spotted as a ~38min stretch (04:33-05:11 on the
2026-08-28 night) that visually and statistically didn't match the rest of
that night's deep sleep: HR swung by 5-9 bpm sample-to-sample, quasi-periodic
on roughly a 4-9 min cycle (weak autocorrelation peak, lag ~7-8 min), with
movement staying flat throughout - so neither the movement-gated REM
classifier nor the joint movement+HR arousal detector (which requires a
*movement* response, not just an HR one) ever fired on it.

Three competing hypotheses, none confirmable from this recording's signals
(accelerometer + HR only - no SpO2, no respiration/airflow):

- **(a) NREM cyclic alternating pattern (CAP) / recurring microarousals** - a
  normal autonomic microstructure of light/deep NREM sleep that modulates HR
  cyclically over minutes without producing a movement response. Would argue
  this is legitimately NREM (deep or light), just not the quiescent kind the
  current deep cluster is calibrated on.
- **(b) Cyclic variation of heart rate (CVHR) from repeated brief respiratory
  events (apnea/hypopnea)** - the classic surge-recover HR pattern
  accompanying sleep-disordered breathing. Also leaves no actigraphy
  movement signature, so is equally consistent with the data. Would argue
  for flagging it as a distinct clinically-relevant regime, not a sleep
  stage at all.
- **(c) Under-detected REM** - given the night's aggregate REM% is already
  suspiciously low relative to the normal ~20-25% of TST (see the REM
  bout-duration issue above and the aggregate-percentage caveat it's tied
  to), a real possibility is that some/most of `unknown1` actually *is* REM
  that the movement+HR-cluster gate is still failing to recognize - e.g.
  because this particular REM is unusually low-HRV for this person, or
  because the "still" tier's REM cluster center is itself pulled toward the
  small number of extreme HR spikes elsewhere in the night, making it too
  strict to catch quieter REM. This is the interpretation we currently lean
  towards, precisely because it would resolve two anomalies (a too-low REM%
  and this unexplained regime) with one cause - but it is not confirmed, and
  auto-labeling `unknown1` as `rem` would be assuming the conclusion instead
  of testing it.

**To do:** get a second signal (SpO2 minimum, or a respiration-derived
signal) for at least one more `unknown1` night to adjudicate between (a)/(b)
(respiratory events) vs (c) (REM). Until then, keep `unknown1` a separate,
non-committal label rather than folding it into `deep` or `rem` - the
aggregate REM%/deep% numbers already can't be trusted at face value (see
above), and quietly reassigning `unknown1` to `rem` would inflate REM% based
on a hypothesis, not evidence.

## "deep" is 53-76% of sleep on every night

`deep` = still tier + HR nearest the calibrated deep center. N2 is also still
with a calm HR, and the 30-min rolling-median HR baseline is computed mostly
from those same epochs, so `hr_detrend` of the deep cluster is ~0 by
construction. Adult N3 is ~13-23% of sleep; the Zeblaze watch reported
23-30% on 2026-08/09 nights. The report now warns outside 10-35%, which so
far fires on all 9 recorded nights.

**To do:** find a feature that separates N3 from N2 (candidates: HR below a
night-wide low quantile instead of a local median; N3 concentrated in the
first third of the night; minimum bout length), and validate it against a
reference with staging (Zeblaze nights overlap only 2026-09-01 so far).

## Sleep onset ignores HR

Onset is movement only (first 10 min of immobility). On 2026-11-17 it lands
at 22:55 while HR stays at 60-61 bpm, above its 56 bpm night level, until
23:12. A "HR settled near its night median" condition could tighten it.
