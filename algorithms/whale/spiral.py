"""
=========================================================
Bubble-Net Spiral Attack
=========================================================
"""

from __future__ import annotations

import numpy as np


class SpiralAttack:

    @staticmethod
    def update(current, best):

        b = 1.0

        l = np.random.uniform(-1, 1)

        distance = np.abs(best - current)

        return (

            distance

            * np.exp(b * l)

            * np.cos(2 * np.pi * l)

            + best

        )