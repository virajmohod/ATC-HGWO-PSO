"""
=========================================================
Binary Position Update
=========================================================
"""

from __future__ import annotations

import numpy as np


class WhalePosition:

    @staticmethod
    def sigmoid(x):

        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def binary(vector):

        probability = WhalePosition.sigmoid(vector)

        return (

            np.random.rand(len(vector))

            < probability

        ).astype(int)

    @staticmethod
    def repair(mask):

        if np.sum(mask) == 0:

            mask[np.random.randint(len(mask))] = 1

        return mask