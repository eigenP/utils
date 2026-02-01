
import unittest
import numpy as np
from eigenp_utils.maxproj_registration import apply_drift_correction_2D, estimate_drift_2D, _get_weight_profiles, _2D_weighted_image

class TestDriftWindowingBias(unittest.TestCase):
    """
    Testr 🔎 Verification: Windowing Bias & Orthogonality

    This test suite exposes a subtle flaw in the original drift correction algorithm:
    Applying 2D windowing BEFORE max-projection couples the axes.
    Movement along Y modulates the intensity of the X-projection (via the Y-window),
    breaking the orthogonality of the projections.
    """

    def generate_orthogonal_motion(self, shape=(20, 100, 100)):
        """
        Generates a video where a small object moves ONLY in Y.
        It starts at the very top edge (deep inside the taper).
        """
        T, H, W = shape
        video = np.zeros(shape, dtype=np.float32)

        # Small square moving in Y
        width = 5
        start_y = 0
        x_pos = 50

        for t in range(T):
            # Move Y
            y = start_y + t * 2
            # Clip
            if y + width < H:
                video[t, y:y+width, x_pos:x_pos+width] = 1.0

        return video

    def test_projection_orthogonality(self):
        """
        Verifies that the X-projection energy is independent of Y-position.

        Current Implementation (Flawed): Fails because Y-windowing dimms the object as it moves.
        Fixed Implementation: Should Pass (Constant Energy).
        """
        video = self.generate_orthogonal_motion()
        T, H, W = video.shape

        # OLD LOGIC: 2D Weighting -> Max
        overlap = min(H, W) // 3
        profiles = _get_weight_profiles((H, W), overlap)

        energies_old = []
        energies_new = []

        for t in range(T):
            frame = video[t]

            # --- OLD PATH ---
            # Apply 2D weight
            w_frame = _2D_weighted_image(frame, overlap, profiles=profiles)
            # Max proj X
            proj_x_old = np.max(w_frame, axis=0)
            energies_old.append(np.sum(proj_x_old))

            # --- NEW PATH ---
            # Max proj X first
            raw_proj_x = np.max(frame, axis=0)
            # Apply 1D weight X (profiles[1] is profile_x)
            proj_x_new = raw_proj_x * profiles[1]
            energies_new.append(np.sum(proj_x_new))

        energies_old = np.array(energies_old)
        energies_new = np.array(energies_new)

        # Normalize to max (since it starts at 0 energy in old path)
        norm_old = energies_old / energies_old.max()
        norm_new = energies_new / energies_new.max()

        print(f"\nEnergy Stability (Min/Max Ratio):")
        print(f"Old (2D Window): {norm_old.min():.4f}")
        print(f"New (1D Window): {norm_new.min():.4f}")

        # The OLD logic should show drastic drop (near 0 at start)
        self.assertLess(norm_old.min(), 0.5, "Old logic should show severe attenuation at edges")

        # The NEW logic should be stable (1.0) because X-projection is invariant to Y-pos
        self.assertGreater(norm_new.min(), 0.99, "New logic should be orthogonal (stable energy)")

    def test_integer_input_safety(self):
        """
        Regression Test: Ensure estimate_drift_2D works with integer inputs.
        Previous implementation used in-place multiplication (*=) which crashes on int * float.
        """
        # Create integer frames
        frame1 = np.ones((100, 100), dtype=np.uint8)
        frame2 = np.ones((100, 100), dtype=np.uint8)

        # Shift frame2 slightly (conceptually)
        # But even with identity, it should run without error.

        try:
            drift = estimate_drift_2D(frame1, frame2)
            print(f"\nInteger Input Test: Drift {drift}")
        except Exception as e:
            self.fail(f"estimate_drift_2D failed on integer input: {e}")

if __name__ == '__main__':
    unittest.main()
