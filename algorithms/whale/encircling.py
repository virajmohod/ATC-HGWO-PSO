"""
=========================================================
Encircling Prey Operator
=========================================================
"""

from __future__ import annotations

import numpy as np


class Encircling:

    @staticmethod
    def coefficient(iteration, max_iteration):

        a = 2 - 2 * iteration / max_iteration

        r1 = np.random.rand()

        r2 = np.random.rand()

        A = 2 * a * r1 - a

        C = 2 * r2

        return A, C

    @staticmethod
    def update(current, best, iteration, max_iteration):

        A, C = Encircling.coefficient(
            iteration,
            max_iteration,
        )

        D = np.abs(C * best - current)

        X = best - A * D

        return X