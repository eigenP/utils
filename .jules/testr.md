# Testr Journal

## 2025-02-18 - Drift Correction Integrator Windup
**Learning:** The drift correction algorithm in `maxproj_registration.py` was casting cumulative drift to integer at every time step (`cum_dx = int(cum_dx + dx)`). This introduced a severe "integrator windup" bug (or rather, failure to integrate) where fractional drifts smaller than 1.0 would be discarded repeatedly if they didn't cross an integer boundary in a single step. For slow drifts (e.g., 0.5 px/frame), this resulted in zero correction over time.
**Action:** The accumulation logic was fixed to maintain float precision (`cum_dx += dx`) and only round to integer when applying the final shift. A high-level invariant test (`tests/test_drift_integrity.py`) was added to verify that fractional drift accumulates correctly over time (Linearity/conservation of total drift).
