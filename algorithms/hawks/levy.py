"""
=========================================================
Levy Flight
=========================================================
"""

from __future__ import annotations

import numpy as np
from math import gamma, sin, pi


class LevyFlight:

    @staticmethod
    def generate(dimension):

        beta = 1.5

        sigma = (

            gamma(1 + beta)

            * np.sin(np.pi * beta / 2)

            /

            (

                gamma((1 + beta) / 2)

                * beta

                * 2 ** ((beta - 1) / 2)

            )

        ) ** (1 / beta)

        u = np.random.normal(

            0,

            sigma,

            dimension,

        )

        v = np.random.normal(

            0,

            1,

            dimension,

        )

        step = u / (

            np.abs(v) ** (1 / beta)

        )

        return step