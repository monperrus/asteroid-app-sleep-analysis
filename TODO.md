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
