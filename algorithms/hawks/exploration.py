"""
=========================================================
Exploration Phase
=========================================================
"""

from __future__ import annotations

import numpy as np


class Exploration:

    @staticmethod
    def random_search(

        current,

        random_hawk,

        mean_position,

        energy,

    ):

        q = np.random.rand()

        if q >= 0.5:

            return (

                random_hawk

                - np.random.rand()

                * np.abs(

                    random_hawk

                    - 2 * np.random.rand() * current

                )

            )

        return (

            mean_position

            - energy

            * np.abs(

                np.random.rand()

                * mean_position

                - current

            )

        )