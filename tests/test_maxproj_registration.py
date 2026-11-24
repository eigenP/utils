import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import center_of_mass, gaussian_filter, shift
from scipy.signal import windows
from skimage import data, transform

from eigenp_utils.maxproj_registration import (
    apply_drift_correction,
    apply_subpixel_drift_correction,
    compute_drift_trajectory,
    estimate_drift,
    estimate_shift_1d_iterative,
    zero_shift_multi_dimensional,
)


# --- Shared synthetic generators ---


def _make_drifting_gaussian(n_frames=20, shape=(64, 64), drift_per_frame=(0.0, 1.0), sigma=3.0):
    """Generate a synthetic 2D+t video (T, H, W) of a drifting Gaussian blob and its centroids."""
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2.0, W / 2.0
    dy, dx = drift_per_frame

    frames = np.zeros((n_frames, H, W), dtype=np.float32)
    gt_positions = np.zeros((n_frames, 2), dtype=np.float64)
    for t in range(n_frames):
        curr_cy = cy + dy * t
        curr_cx = cx + dx * t
        gt_positions[t] = (curr_cy, curr_cx)
        frames[t] = np.exp(-((y - curr_cy) ** 2 + (x - curr_cx) ** 2) / (2 * sigma**2))

    return frames, gt_positions


def _measure_centroid_rmse(video_data):
    """Return RMSE of frame centroids relative to their temporal mean position."""
    centroids = np.array([center_of_mass(frame) for frame in video_data])
    deviations = centroids - np.mean(centroids, axis=0)
    return float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))


def _create_square_frame_noisy(shape, object_pos, object_size, intensity=1.0, noise_level=0.1, seed=42):
    """Generate a 2D noisy frame containing a bright square at object_pos."""
    rng = np.random.default_rng(seed)
    frame = rng.normal(0.0, noise_level, shape).astype(np.float32)
    y, x = object_pos
    h, w = object_size
    y_start, y_end = max(0, int(y - h // 2)), min(shape[0], int(y + h // 2))
    x_start, x_end = max(0, int(x - w // 2)), min(shape[1], int(x + w // 2))
    frame[y_start:y_end, x_start:x_end] += intensity
    return frame


def _two_peak_profile(n=200):
    """Smooth 1D asymmetric two-peak profile for drift-estimation tests."""
    x = np.arange(n)
    return np.exp(-0.5 * ((x - 80) / 4) ** 2) * 1000 + np.exp(-0.5 * ((x - 130) / 6) ** 2) * 600


def _blob_scene(H=160, W=160, n=12, seed=3, sigma=2.5):
    """Normalized 2D scene with randomly placed Gaussian blobs."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W))
    for _ in range(n):
        img[int(rng.uniform(30, H - 30)), int(rng.uniform(30, W - 30))] = 1.0
    img = gaussian_filter(img, sigma)
    return img / img.max()


@pytest.fixture(scope="module")
def oscillating_run():
    """Run bidirectional subpixel drift correction once on a 50-frame oscillating video."""
    n_cycles, period, amplitude, size = 5, 10, 3.0, 256
    try:
        scene = data.camera()
    except Exception:
        scene = transform.resize(data.checkerboard(), (512, 512), preserve_range=True).astype(np.uint8)
    scene = scene.astype(np.float32)

    H, W = scene.shape
    cy, cx = H // 2, W // 2
    half_sz = size // 2
    n_frames = n_cycles * period
    video = np.zeros((n_frames, size, size), dtype=np.uint8)
    gt_drift = np.zeros((n_frames, 2), dtype=np.float64)

    for t in range(n_frames):
        angle = 2 * np.pi * t / period
        dx = amplitude * np.sin(angle)
        dy = amplitude * np.cos(angle)
        shifted = shift(scene, shift=(dy, dx), order=3, mode="reflect")
        video[t] = shifted[cy - half_sz : cy + half_sz, cx - half_sz : cx + half_sz].astype(np.uint8)
        gt_drift[t] = (dx, dy)

    _, table = apply_drift_correction(video, method="subpixel", reverse_time="both", save_drift_table=False)
    return table, gt_drift, period, n_cycles


# --- zero_shift_multi_dimensional ---


def test_zero_shift_positive_negative():
    """Positive and negative shifts move values accurately and fill exposed edges with fill_value."""
    arr = np.arange(9).reshape(3, 3)
    result = zero_shift_multi_dimensional(arr, shifts=(1, -1), fill_value=-1)
    expected = np.array([
        [-1, -1, -1],
        [1, 2, -1],
        [4, 5, -1],
    ])
    np.testing.assert_array_equal(result, expected)


def test_zero_shift_errors():
    """Mismatched shift dimensionality raises ValueError and non-integer shifts raise TypeError."""
    arr = np.zeros((2, 2))
    with pytest.raises(ValueError):
        zero_shift_multi_dimensional(arr, shifts=(1,))
    with pytest.raises(TypeError):
        zero_shift_multi_dimensional(arr, shifts=(1.0, 2.0))


def test_zero_shift_accepts_numpy_integers():
    """NumPy scalar integers and integer ndarrays are accepted as shift arguments."""
    a = np.arange(12).reshape(3, 4)
    expected = zero_shift_multi_dimensional(a, shifts=(1, 0))
    np.testing.assert_array_equal(zero_shift_multi_dimensional(a, shifts=(np.int64(1), 0)), expected)
    np.testing.assert_array_equal(zero_shift_multi_dimensional(a, shifts=np.array([1, 0])), expected)
    np.testing.assert_array_equal(
        zero_shift_multi_dimensional(a, shifts=np.int64(1)),
        zero_shift_multi_dimensional(a, shifts=1),
    )


def test_zero_shift_out_buffer_and_large_shifts():
    """out= writes in-place into the supplied buffer, and |shift| >= axis length fills entirely."""
    arr = np.arange(12).reshape(3, 4)
    buf = np.full_like(arr, 99)
    ret = zero_shift_multi_dimensional(arr, shifts=(1, -1), fill_value=-1, out=buf)

    assert ret is buf
    expected = np.array([
        [-1, -1, -1, -1],
        [1, 2, 3, -1],
        [5, 6, 7, -1],
    ])
    np.testing.assert_array_equal(buf, expected)

    for large_shift in [(3, 0), (5, 0), (-3, 0), (-5, 0), (0, 4), (0, -6)]:
        out = zero_shift_multi_dimensional(arr, shifts=large_shift, fill_value=-7)
        np.testing.assert_array_equal(out, np.full_like(arr, -7))


# --- apply_drift_correction core & dimension handling ---


def test_apply_drift_correction_2d_plus_t():
    """2D+t integer drift correction recovers exact step/cumulative shifts and aligns every frame."""
    video = np.zeros((3, 10, 10), dtype=np.float32)
    video[0, 2:4, 2:4] = 1.0
    video[1, 3:5, 4:6] = 1.0  # +1 in Y, +2 in X
    video[2, 4:6, 6:8] = 1.0  # +2 in Y, +4 in X

    corrected, drift_table = apply_drift_correction(video, save_drift_table=False)

    expected_cols = ["Time Point", "dx", "dy", "dz", "cum_dx", "cum_dy", "cum_dz"]
    assert list(drift_table.columns) == expected_cols
    np.testing.assert_array_equal(drift_table["dz"].to_numpy(), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(drift_table["cum_dz"].to_numpy(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(drift_table["dy"].to_numpy(), [0.0, -1.0, -1.0], atol=0.05)
    np.testing.assert_allclose(drift_table["dx"].to_numpy(), [0.0, -2.0, -2.0], atol=0.05)
    np.testing.assert_allclose(drift_table["cum_dy"].to_numpy(), [0.0, -1.0, -2.0], atol=0.05)
    np.testing.assert_allclose(drift_table["cum_dx"].to_numpy(), [0.0, -2.0, -4.0], atol=0.05)

    for t in range(3):
        np.testing.assert_array_equal(corrected[t], video[0])


def test_apply_drift_correction_3d_plus_t():
    """3D+t integer drift correction recovers exact Z/Y/X shifts and aligns every volume to frame 0."""
    video = np.zeros((3, 10, 10, 10), dtype=np.float32)
    video[0, 2:4, 2:4, 2:4] = 1.0
    video[1, 3:5, 3:5, 4:6] = 1.0  # +1 in Z, +1 in Y, +2 in X
    video[2, 4:6, 4:6, 6:8] = 1.0  # +2 in Z, +2 in Y, +4 in X

    corrected, drift_table = apply_drift_correction(video, save_drift_table=False)

    expected_cols = ["Time Point", "dx", "dy", "dz", "cum_dx", "cum_dy", "cum_dz"]
    assert list(drift_table.columns) == expected_cols
    np.testing.assert_allclose(drift_table["dz"].to_numpy(), [0.0, -1.0, -1.0], atol=0.05)
    np.testing.assert_allclose(drift_table["dy"].to_numpy(), [0.0, -1.0, -1.0], atol=0.05)
    np.testing.assert_allclose(drift_table["dx"].to_numpy(), [0.0, -2.0, -2.0], atol=0.05)
    np.testing.assert_allclose(drift_table["cum_dz"].to_numpy(), [0.0, -1.0, -2.0], atol=0.05)
    np.testing.assert_allclose(drift_table["cum_dy"].to_numpy(), [0.0, -1.0, -2.0], atol=0.05)
    np.testing.assert_allclose(drift_table["cum_dx"].to_numpy(), [0.0, -2.0, -4.0], atol=0.05)

    for t in range(3):
        np.testing.assert_array_equal(corrected[t], video[0])


def test_apply_drift_correction_invalid_dims():
    """Inputs that are neither 3D (T, Y, X) nor 4D (T, Z, Y, X) raise ValueError."""
    with pytest.raises(ValueError, match=r"Expected 3D \(T, Y, X\) or 4D \(T, Z, Y, X\) data"):
        apply_drift_correction(np.zeros((10, 10)))
    with pytest.raises(ValueError, match=r"Expected 3D \(T, Y, X\) or 4D \(T, Z, Y, X\) data"):
        apply_drift_correction(np.zeros((2, 2, 10, 10, 10)))


def test_apply_drift_correction_4d_subpixel():
    """4D (T, Z, Y, X) subpixel registration recovers fractional 3D drift and stabilizes centroids."""
    T, Z, Y, X = 6, 24, 32, 32
    step = (0.35, -0.45, 0.55)
    zz, yy, xx = np.mgrid[:Z, :Y, :X]
    cz, cy, cx = Z / 2.0, Y / 2.0, X / 2.0
    sigma = 2.5

    video = np.zeros((T, Z, Y, X), dtype=np.float32)
    for t in range(T):
        video[t] = np.exp(
            -((zz - (cz + step[0] * t)) ** 2 + (yy - (cy + step[1] * t)) ** 2 + (xx - (cx + step[2] * t)) ** 2)
            / (2 * sigma**2)
        )

    corrected, table = apply_drift_correction(video, method="subpixel", save_drift_table=False)

    assert corrected.shape == video.shape
    assert corrected.dtype == video.dtype
    assert table["cum_dz"].iloc[-1] == pytest.approx(-(T - 1) * step[0], abs=0.3)
    assert table["cum_dy"].iloc[-1] == pytest.approx(-(T - 1) * step[1], abs=0.3)
    assert table["cum_dx"].iloc[-1] == pytest.approx(-(T - 1) * step[2], abs=0.3)
    assert _measure_centroid_rmse(corrected) < 0.1


def test_apply_drift_correction_save_csv_roundtrip(tmp_path):
    """save_drift_table=True writes a CSV file that round-trips back to the returned DataFrame."""
    video, _ = _make_drifting_gaussian(n_frames=6, shape=(64, 64), drift_per_frame=(0.2, 0.5))
    csv_path = tmp_path / "drift_table.csv"

    _, table = apply_drift_correction(video, save_drift_table=True, csv_filename=str(csv_path))

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == list(table.columns)
    np.testing.assert_allclose(loaded.to_numpy(), table.to_numpy(), atol=1e-5)


# --- Oscillation bias, sign conventions & subpixel stability ---


def test_zero_mean_oscillation(oscillating_run):
    """After full cycles of sinusoidal motion, cumulative drift returns near zero (< 1.0 px)."""
    table, _, period, n_cycles = oscillating_run
    cycle_indices = [i * period for i in range(1, n_cycles)]

    max_endpoint_error = 0.0
    for t in cycle_indices:
        row = table[table["Time Point"] == t]
        assert not row.empty, f"Time Point {t} missing from drift table"
        max_endpoint_error = max(
            max_endpoint_error,
            abs(float(row["cum_dx"].iloc[0])),
            abs(float(row["cum_dy"].iloc[0])),
        )

    assert max_endpoint_error < 1.0, (
        f"Significant drift bias detected: max cycle endpoint error {max_endpoint_error:.4f} px"
    )


def test_amplitude_accuracy(oscillating_run):
    """Estimated peak-to-peak drift amplitude matches the sinusoidal ground-truth range within 15%."""
    table, gt_drift, _, _ = oscillating_run
    range_est = float(np.ptp(table["cum_dx"].to_numpy()))
    range_gt = float(np.ptp(gt_drift[:, 0]))
    rel_error = abs(range_est - range_gt) / range_gt

    assert rel_error < 0.15, f"Drift amplitude relative error {rel_error:.1%} exceeds 15%"


@pytest.mark.parametrize("reverse_time", [False, True, "both"])
def test_correction_opposes_drift_across_modes(reverse_time):
    """Forward, reverse, and bidirectional modes oppose +X drift with the correct reference sign."""
    n_frames = 10
    video, _ = _make_drifting_gaussian(n_frames=n_frames, shape=(64, 64), drift_per_frame=(0.0, 1.0))

    corrected, table = apply_drift_correction(video, reverse_time=reverse_time, save_drift_table=False)

    if reverse_time is True:
        # Reverse mode anchors at the final frame (t=T-1) and shifts earlier frames forward (+X)
        assert table["cum_dx"].iloc[-1] == 0.0
        assert table["cum_dx"].iloc[0] == pytest.approx(n_frames - 1, abs=1.0)
    else:
        # Forward and bidirectional modes anchor at t=0 and shift later frames backward (-X)
        assert table["cum_dx"].iloc[0] == 0.0
        assert table["cum_dx"].iloc[-1] < -5.0

    c_first = center_of_mass(corrected[0])
    c_last = center_of_mass(corrected[-1])
    assert abs(c_last[1] - c_first[1]) < 2.5


def test_subpixel_vs_integer_precision():
    """Subpixel correction reduces centroid jitter by >5x relative to integer snapping."""
    video, _ = _make_drifting_gaussian(n_frames=30, shape=(128, 128), drift_per_frame=(0.33, 0.33))

    corrected_int, _ = apply_drift_correction(video.copy(), method="integer", save_drift_table=False)
    corrected_sub, _ = apply_drift_correction(video.copy(), method="subpixel", save_drift_table=False)

    rmse_int = _measure_centroid_rmse(corrected_int)
    rmse_sub = _measure_centroid_rmse(corrected_sub)

    assert rmse_sub < 0.1, f"Subpixel RMSE {rmse_sub:.4f} exceeded 0.1 px"
    assert rmse_sub < 0.2 * rmse_int, f"Subpixel ({rmse_sub:.4f}) not >5x better than integer ({rmse_int:.4f})"


def test_cycle_consistency_stationary():
    """Subpixel correction on a stationary video detects ~0 drift and preserves the image."""
    video, _ = _make_drifting_gaussian(n_frames=10, shape=(64, 64), drift_per_frame=(0.0, 0.0))

    corrected, table = apply_drift_correction(video.copy(), method="subpixel")

    max_cum_drift = float(np.max(np.abs(table[["cum_dx", "cum_dy"]].to_numpy())))
    assert max_cum_drift < 0.05
    assert float(np.mean((corrected - video) ** 2)) < 1e-6


def test_cycle_consistency_exact():
    """Correcting a known (0.5, 0.5) bicubic-shifted frame restores the reference with MSE < 1e-5."""
    H, W = 64, 64
    y, x = np.mgrid[:H, :W]
    frame0 = np.exp(-((y - H // 2) ** 2 + (x - W // 2) ** 2) / (2 * 4.0**2)).astype(np.float32)

    drift = (0.5, 0.5)
    frame1 = shift(frame0, drift, order=3, mode="constant", cval=0.0)
    video = np.array([frame0, frame1])

    corrected, table = apply_drift_correction(video, method="subpixel", save_drift_table=False)

    assert table["cum_dy"].iloc[-1] == pytest.approx(-drift[0], abs=0.05)
    assert table["cum_dx"].iloc[-1] == pytest.approx(-drift[1], abs=0.05)
    assert float(np.mean((corrected[1] - frame0) ** 2)) < 1e-5


def test_apply_subpixel_drift_correction_2d_and_3d():
    """apply_subpixel_drift_correction fills borders with image.min(), shifts 3D, and round-trips."""
    # 2D image with a non-zero baseline offset
    H, W = 64, 64
    y, x = np.mgrid[:H, :W]
    img_2d = (np.exp(-((y - 32) ** 2 + (x - 32) ** 2) / (2 * 4.0**2)) * 50.0 + 10.0).astype(np.float32)

    shifted = apply_subpixel_drift_correction(img_2d, (2.3, -1.7))
    assert shifted.shape == img_2d.shape
    np.testing.assert_allclose(shifted[:2, :], img_2d.min(), atol=1e-5)
    np.testing.assert_allclose(shifted[:, -1:], img_2d.min(), atol=1e-5)

    # Cycle consistency in the interior
    restored = apply_subpixel_drift_correction(shifted, (-2.3, 1.7))
    assert float(np.mean((restored[8:-8, 8:-8] - img_2d[8:-8, 8:-8]) ** 2)) < 1e-3

    # 3D volume shift moves the 3D centroid accurately and fills exposed Z border with vol.min()
    zz, yy, xx = np.mgrid[:16, :24, :24]
    vol_3d = np.exp(-((zz - 8) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 2.2**2)).astype(np.float32)
    drift_3d = (1.25, -1.5, 0.75)
    shifted_3d = apply_subpixel_drift_correction(vol_3d, drift_3d)

    assert shifted_3d.shape == vol_3d.shape
    np.testing.assert_allclose(shifted_3d[0, :, :], vol_3d.min(), atol=1e-5)
    c_before = np.array(center_of_mass(vol_3d))
    c_after = np.array(center_of_mass(np.clip(shifted_3d, 0.0, None)))
    np.testing.assert_allclose(c_after - c_before, drift_3d, atol=0.1)


# --- Edge robustness, accumulation & accuracy ---


def test_drift_edge_object():
    """Decoupled 1D windowing accurately estimates X-shift for an object near the top Y-edge."""
    shape = (128, 128)
    frame1 = _create_square_frame_noisy(shape, (10, 64), (10, 10), intensity=1.0, noise_level=0.2)
    frame2 = _create_square_frame_noisy(shape, (10, 84), (10, 10), intensity=1.0, noise_level=0.2)

    est = estimate_drift(frame1, frame2)
    np.testing.assert_allclose(est, np.array([-20.0, 0.0]), atol=1.0)


def test_estimate_drift_return_ccm():
    """estimate_drift(return_ccm=True) returns identical shift and a finite 2D cross-correlation map."""
    base = _blob_scene(H=64, W=64, sigma=3.0)
    f1 = base * 200.0
    f2 = shift(base, (3.0, -2.0), order=3, mode="nearest") * 200.0

    shift_only = estimate_drift(f1, f2, return_ccm=False)
    shift_with_ccm, ccm = estimate_drift(f1, f2, return_ccm=True)

    np.testing.assert_array_equal(shift_with_ccm, shift_only)
    assert ccm.shape == (2, f1.shape[0])
    assert np.isrealobj(ccm)
    assert np.all(np.isfinite(ccm))
    assert float(ccm.max()) > float(ccm.mean())


def test_apply_drift_correction_edge():
    """apply_drift_correction tracks a sequence of 10 px X-shifts for an edge-located object."""
    shape = (128, 128)
    video = np.zeros((3, *shape), dtype=np.float32)
    for t, s in enumerate([0, 10, 20]):
        video[t] = _create_square_frame_noisy(shape, (10, 64 + s), (10, 10), intensity=1.0, noise_level=0.1)

    _, table = apply_drift_correction(video, save_drift_table=False)

    dx_vals = table["dx"].to_numpy()
    assert len(dx_vals) == 3
    np.testing.assert_allclose(dx_vals[1:], -10.0, atol=1.0)


def test_fractional_drift_accumulation():
    """compute_drift_trajectory accumulates sub-pixel shifts in float without premature int truncation."""
    prof = _two_peak_profile(200)
    n_frames = 20
    step_y, step_x = 0.25, 0.35
    projs_y = np.stack([shift(prof, step_y * t, order=3, mode="nearest") for t in range(n_frames)])
    projs_x = np.stack([shift(prof, step_x * t, order=3, mode="nearest") for t in range(n_frames)])

    table = compute_drift_trajectory([projs_y, projs_x])

    np.testing.assert_allclose(table["dy"].to_numpy()[1:], -step_y, atol=0.05)
    np.testing.assert_allclose(table["dx"].to_numpy()[1:], -step_x, atol=0.05)
    np.testing.assert_allclose(table["cum_dy"].to_numpy(), np.cumsum(table["dy"].to_numpy()), atol=1e-5)
    np.testing.assert_allclose(table["cum_dx"].to_numpy(), np.cumsum(table["dx"].to_numpy()), atol=1e-5)
    assert table["cum_dy"].iloc[-1] == pytest.approx(-(n_frames - 1) * step_y, abs=0.25)
    assert table["cum_dx"].iloc[-1] == pytest.approx(-(n_frames - 1) * step_x, abs=0.25)


def test_subpixel_drift_recovery():
    """End-to-end integer drift correction recovers fractional cumulative drift within +/-0.7 px jitter."""
    n_frames = 20
    drift_rate = (0.3, 0.4)
    video, _ = _make_drifting_gaussian(
        n_frames=n_frames, shape=(64, 64), drift_per_frame=drift_rate, sigma=2.5
    )

    corrected_video, drift_table = apply_drift_correction(video, save_drift_table=False)

    assert drift_table["cum_dy"].iloc[-1] == pytest.approx(-(n_frames - 1) * drift_rate[0], abs=0.5)
    assert drift_table["cum_dx"].iloc[-1] == pytest.approx(-(n_frames - 1) * drift_rate[1], abs=0.5)

    centroids = np.array([center_of_mass(frame) for frame in corrected_video])
    assert float(np.max(np.abs(centroids[:, 0] - centroids[0, 0]))) < 0.7
    assert float(np.max(np.abs(centroids[:, 1] - centroids[0, 1]))) < 0.7


# --- Regressions & edge cases ---


@pytest.mark.parametrize("baseline", [0.0, 500.0, 5000.0])
def test_shift_estimate_independent_of_background_offset(baseline):
    """A constant pedestal must not anchor the correlation peak at zero shift."""
    prof = _two_peak_profile()
    win = windows.tukey(prof.size, alpha=0.1)
    mov = shift(prof, 6.7, order=3, mode="nearest")
    est = estimate_shift_1d_iterative(prof + baseline, mov + baseline, win)
    assert est == pytest.approx(-6.7, abs=0.05)
    assert abs(estimate_shift_1d_iterative(prof + baseline, prof + baseline, win)) < 1e-3


def test_drift_correction_with_camera_offset():
    """Camera offset must not pull cumulative drift towards zero for float or Poisson uint16 videos."""
    T, step = 6, np.array([1.5, -2.0])
    truth = -step[None, :] * np.arange(T)[:, None]

    base = _blob_scene(sigma=6.0)
    for offset in (0, 1000):
        video = np.stack(
            [shift(base, step * t, order=3, mode="nearest") * 300 + offset for t in range(T)]
        ).astype(np.float32)
        _, table = apply_drift_correction(video)
        assert np.abs(table[["cum_dy", "cum_dx"]].to_numpy() - truth).max() < 0.1, offset

    base = _blob_scene()
    for seed in range(3):
        frames = np.stack([shift(base, step * t, order=3, mode="nearest") * 200 + 1000 for t in range(T)])
        video = np.random.default_rng(seed).poisson(np.clip(frames, 0, None)).astype(np.uint16)
        _, table = apply_drift_correction(video)
        assert np.abs(table[["cum_dy", "cum_dx"]].to_numpy() - truth).max() < 1.0, seed

    static = np.stack([base * 300 + 1000] * 4).astype(np.float32)
    _, table = apply_drift_correction(static)
    assert np.abs(table[["cum_dy", "cum_dx"]].to_numpy()).max() < 0.05


def test_estimate_drift_with_camera_offset():
    """estimate_drift remains invariant to an additive camera offset of 1000."""
    base = _blob_scene(sigma=6.0)
    f1 = base * 300
    f2 = shift(base, (4.0, -3.0), order=3, mode="nearest") * 300
    for offset in (0, 1000):
        np.testing.assert_allclose(estimate_drift(f1 + offset, f2 + offset), [3.0, -4.0], atol=0.1)


def test_compute_drift_trajectory_defaults_and_short_axes():
    """windows=None works, short axes (limit // 5 == 0) still correct 1 px, and large jumps clamp to 0."""
    prof = _two_peak_profile()
    projs = np.stack([shift(prof, 0.5 * t, order=3, mode="nearest") for t in range(5)])

    table = compute_drift_trajectory([projs])
    np.testing.assert_allclose(table["dx"].to_numpy()[1:], -0.5, atol=0.05)

    table_short = compute_drift_trajectory([projs], shape_limit=(4,))
    np.testing.assert_allclose(table_short["dx"].to_numpy()[1:], -0.5, atol=0.05)

    table_jump = compute_drift_trajectory(
        [np.stack([prof, shift(prof, 7.0, order=3, mode="nearest")])], shape_limit=(30,)
    )
    assert table_jump["dx"].iloc[1] == 0.0


def test_subpixel_correction_rounds_integer_output():
    """Storing bicubic-shifted frames into uint16 rounds to nearest integer rather than truncating."""
    base = _blob_scene(H=96, W=96, n=8)
    rng = np.random.default_rng(0)
    video = np.stack(
        [rng.poisson(shift(base, (0.37 * t, -0.61 * t), order=3, mode="nearest") * 300 + 100) for t in range(4)]
    ).astype(np.uint16)
    corrected, table = apply_drift_correction(video, method="subpixel")
    t = 3
    expected = shift(
        video[t].astype(np.float32),
        (table.loc[t, "cum_dy"], table.loc[t, "cum_dx"]),
        order=3,
        mode="constant",
        cval=video.min(),
    )
    diff = corrected[t].astype(np.float64) - expected
    inner = diff[10:-10, 10:-10]
    assert abs(inner.mean()) < 0.05
    assert np.abs(inner).max() <= 0.5 + 1e-3


def test_subpixel_integer_clamps_bicubic_overshoot_and_undershoot():
    """Bicubic ringing on a sharp 0/255 step edge is clamped to [0, 255] without uint8 wrap-around."""
    frame1 = np.zeros((64, 64), dtype=np.uint8)
    frame1[20:44, 20:44] = 255
    # Frame 0 is bilinear-shifted by (-0.5, -0.5) so frame 1 needs a fractional (-0.5, -0.5) bicubic shift
    frame0 = np.clip(
        np.rint(shift(frame1.astype(np.float32), (-0.5, -0.5), order=1, mode="constant", cval=0.0)),
        0,
        255,
    ).astype(np.uint8)
    video = np.stack([frame0, frame1], axis=0)

    corrected, table = apply_drift_correction(video, method="subpixel", save_drift_table=False)

    assert corrected.dtype == np.uint8
    assert table["cum_dy"].iloc[1] == pytest.approx(-0.5, abs=0.1)
    assert table["cum_dx"].iloc[1] == pytest.approx(-0.5, abs=0.1)
    # Without clamping, undershoot (<0) would wrap to ~240 outside and overshoot (>255) to ~15 inside
    assert int(corrected[1, :15, :].max()) < 5
    assert int(corrected[1, 17:19, 24:40].max()) < 20
    assert int(corrected[1, 22:42, 22:42].min()) > 235


@pytest.mark.parametrize("method", ["integer", "subpixel"])
def test_single_frame_and_constant_video(method):
    """T=1 and spatially constant videos produce zero drift, no NaNs, and unchanged frames."""
    rng = np.random.default_rng(21)
    single = rng.uniform(0.0, 10.0, size=(1, 16, 16)).astype(np.float32)
    corr_single, table_single = apply_drift_correction(single, method=method)

    assert corr_single.shape == single.shape
    assert len(table_single) == 1
    np.testing.assert_array_equal(table_single[["dx", "dy", "dz", "cum_dx", "cum_dy", "cum_dz"]].to_numpy(), 0.0)
    np.testing.assert_allclose(corr_single, single, atol=1e-6)

    const = np.full((5, 32, 32), 42.0, dtype=np.float32)
    corr_const, table_const = apply_drift_correction(const, method=method)

    assert np.all(np.isfinite(table_const.to_numpy()))
    np.testing.assert_array_equal(table_const[["dx", "dy", "dz", "cum_dx", "cum_dy", "cum_dz"]].to_numpy(), 0.0)
    np.testing.assert_array_equal(corr_const, const)
