
import unittest
import numpy as np
import matplotlib.colors
from eigenp_utils.tnia_plotting_3d import create_multichannel_rgb

class TestColorBlendingProperties(unittest.TestCase):
    """
    Testr 🔎 Verification: Color Blending Integrity

    Verifies the mathematical correctness of multichannel image blending,
    specifically focusing on:
    1. Soft Clipping (Hue Preservation): Ensuring that over-saturated pixels
       scale down to preserve channel ratios (color fidelity) rather than
       clipping independently (which artifacts towards yellow/white).
    2. Blending Modes: Verifying 'screen' blending math ($1 - (1-A)(1-B)$).
    3. Gamma Correction: Verifying non-linear tone mapping application.
    """

    def test_hue_preservation_soft_clip(self):
        """
        Verify that `soft_clip=True` preserves the ratio between channels (Hue)
        when the sum exceeds 1.0.

        Scenario:
        - Channel 1 (Red): 0.8
        - Channel 2 (Orange): 0.8 (Orange = 1.0 Red, 0.5 Green)

        Sum:
        - Red: 0.8 + 0.8 * 1.0 = 1.6
        - Green: 0 + 0.8 * 0.5 = 0.4

        Soft Clip (Preserve Ratio):
        - Max is 1.6. Scale by 1.6.
        - Red: 1.6 / 1.6 = 1.0
        - Green: 0.4 / 1.6 = 0.25

        Hard Clip (Destroy Ratio):
        - Red: min(1.6, 1.0) = 1.0
        - Green: min(0.4, 1.0) = 0.4

        Difference: 0.25 vs 0.4. Significant hue shift.
        """
        # 1x1 pixel image
        shape = (1, 1)

        # Inputs within range [0, 1] but summing to > 1
        ch1 = np.full(shape, 0.8)
        ch2 = np.full(shape, 0.8)

        # Dummy lists
        xy_list = [ch1, ch2]
        xz_list = [ch1, ch2]
        zy_list = [ch1, ch2]

        # Colors: Red and Orange
        colors = [(1, 0, 0), (1, 0.5, 0)]

        # Run with Soft Clip (Default)
        xy_soft, _, _ = create_multichannel_rgb(
            xy_list, xz_list, zy_list,
            colors=colors,
            blend='add',
            soft_clip=True,
            vmin=0, vmax=1
        )

        pixel = xy_soft[0, 0] # [R, G, B]

        print(f"\nSoft Clip Input: Red=0.8, Orange=0.8")
        print(f"Soft Clip Output: {pixel}")

        # Expected: Red=1.0, Green=0.25
        self.assertAlmostEqual(pixel[0], 1.0, delta=1e-5, msg="Red should be maxed at 1.0")
        self.assertAlmostEqual(pixel[1], 0.25, delta=1e-5, msg="Green should be 0.25 (Ratio preserved)")

        # Run with Hard Clip
        xy_hard, _, _ = create_multichannel_rgb(
            xy_list, xz_list, zy_list,
            colors=colors,
            blend='add',
            soft_clip=False,
            vmin=0, vmax=1
        )

        pixel_h = xy_hard[0, 0]
        print(f"Hard Clip Output: {pixel_h}")

        # Expected: Red=1.0, Green=0.4
        self.assertAlmostEqual(pixel_h[0], 1.0, delta=1e-5)
        self.assertAlmostEqual(pixel_h[1], 0.4, delta=1e-5, msg="Hard clip should shift hue (0.4 vs 0.25)")

    def test_screen_blending_math(self):
        """
        Verify 'screen' blending mode: $1 - (1-A)(1-B)$.
        It creates a softer blend than 'add', never exceeding 1.0 naturally.
        """
        shape = (1, 1)
        # Inputs 0.5 and 0.5
        # Norm: vmin=0, vmax=1 -> values remain 0.5
        ch1 = np.full(shape, 0.5)
        ch2 = np.full(shape, 0.5)

        # If we use Red and Red, we get pure Red channel blending.
        # 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75.

        xy_list = [ch1, ch2]
        xz_list = [ch1, ch2]
        zy_list = [ch1, ch2]

        colors = ['red', 'red'] # Both red to blend in same channel

        xy_screen, _, _ = create_multichannel_rgb(
            xy_list, xz_list, zy_list,
            colors=colors,
            blend='screen',
            vmin=0, vmax=1
        )

        pixel = xy_screen[0, 0]
        print(f"\nScreen Blend (0.5, 0.5): {pixel}")

        expected = 1.0 - (1.0 - 0.5) * (1.0 - 0.5) # 0.75

        self.assertAlmostEqual(pixel[0], expected, delta=1e-5)

    def test_gamma_correction(self):
        """
        Verify Gamma correction curve.
        Input 0.25 (linear). Gamma 0.5 (Sqrt).
        Expected: 0.25^0.5 = 0.5.
        """
        shape = (1, 1)
        ch1 = np.full(shape, 0.25)

        xy_list = [ch1]
        xz_list = [ch1]
        zy_list = [ch1]

        # Red channel
        colors = ['red']

        xy_gamma, _, _ = create_multichannel_rgb(
            xy_list, xz_list, zy_list,
            colors=colors,
            blend='add',
            gamma=0.5, # Brightening
            vmin=0, vmax=1
        )

        pixel = xy_gamma[0, 0]
        print(f"\nGamma 0.5 on 0.25: {pixel}")

        self.assertAlmostEqual(pixel[0], 0.5, delta=1e-5, msg="Gamma correction failed (sqrt(0.25) != 0.5)")

    def test_max_blending(self):
        """
        Verify 'max' blending mode.
        Should take the maximum value per channel.
        """
        shape = (1, 1)
        ch1 = np.full(shape, 0.3)
        ch2 = np.full(shape, 0.8)

        # Both Red
        colors = ['red', 'red']

        xy_max, _, _ = create_multichannel_rgb(
            [ch1, ch2], [ch1, ch2], [ch1, ch2],
            colors=colors,
            blend='max',
            vmin=0, vmax=1
        )

        pixel = xy_max[0, 0]
        print(f"\nMax Blend (0.3, 0.8): {pixel}")

        self.assertAlmostEqual(pixel[0], 0.8, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
