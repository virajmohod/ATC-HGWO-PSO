"""
=========================================================
Tent Chaotic Initialization
=========================================================

Used for generating an initial population with better
diversity than random initialization.

Reference:
Tent Chaotic Mapping

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np


class TentChaoticMap:

    def __init__(

        self,

        population_size,

        dimension,

        mu=0.6,

    ):

        self.population_size = population_size
        self.dimension = dimension
        self.mu = mu

    # ----------------------------------------------------

    def tent(self, x):

        if x < self.mu:

            return x / self.mu

        return (1 - x) / (1 - self.mu)

    # ----------------------------------------------------

    def generate_sequence(self):

        sequence = np.zeros(

            self.population_size *

            self.dimension

        )

        sequence[0] = np.random.rand()

        for i in range(

            1,

            len(sequence),

        ):

            sequence[i] = self.tent(

                sequence[i - 1]

            )

        return sequence

    # ----------------------------------------------------

    def initialize(self):

        chaotic = self.generate_sequence()

        chaotic = chaotic.reshape(

            self.population_size,

            self.dimension,

        )

        population = (

            chaotic >= 0.5

        ).astype(int)

        for i in range(

            self.population_size

        ):

            if np.sum(

                population[i]

            ) == 0:

                population[

                    i,

                    np.random.randint(

                        self.dimension

                    ),

                ] = 1

        return population