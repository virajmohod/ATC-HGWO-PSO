"""
=========================================================
Leader Salp Update
=========================================================
"""

from __future__ import annotations

import numpy as np


class LeaderUpdate:

    @staticmethod
    def coefficient(iteration, max_iteration):

        return 2 * np.exp(
            -(4 * iteration / max_iteration) ** 2
        )

    @staticmethod
    def update(current, food, iteration, max_iteration):

        c1 = LeaderUpdate.coefficient(
            iteration,
            max_iteration,
        )

        c2 = np.random.rand(len(current))
        c3 = np.random.rand(len(current))

        new_position = np.where(
            c3 >= 0.5,
            food + c1 * c2,
            food - c1 * c2,
        )

        return new_position