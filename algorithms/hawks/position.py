"""
=========================================================
Binary Position Update
=========================================================
"""

from __future__ import annotations

import numpy as np


class HawkPosition:

    @staticmethod
    def sigmoid(x):

        return 1 / (

            1

            + np.exp(-x)

        )

    @staticmethod
    def binary(vector):

        p = HawkPosition.sigmoid(vector)

        return (

            np.random.rand(

                len(vector)

            )

            < p

        ).astype(int)

    @staticmethod
    def repair(mask):

        if np.sum(mask) == 0:

            mask[

                np.random.randint(

                    len(mask)

                )

            ] = 1

        return mask