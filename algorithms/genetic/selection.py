"""
=========================================================
Selection Operators
=========================================================

1. Tournament Selection

2. Roulette Wheel Selection

3. Rank Selection

Author : Viraj Mohod
"""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome


class Selection:

    @staticmethod
    def tournament(

        population,

        k: int = 3,

    ) -> Chromosome:

        indices = np.random.choice(

            len(population),

            k,

            replace=False,

        )

        candidates = [

            population[i]

            for i in indices

        ]

        candidates.sort(

            key=lambda c: c.fitness

        )

        return candidates[0].copy()

    @staticmethod
    def roulette(

        population,

    ) -> Chromosome:

        fitness = np.array(

            [

                1 / (c.fitness + 1e-10)

                for c in population

            ]

        )

        probability = fitness / fitness.sum()

        index = np.random.choice(

            len(population),

            p=probability,

        )

        return population[index].copy()

    @staticmethod
    def rank(

        population,

    ) -> Chromosome:

        ranked = sorted(

            population,

            key=lambda c: c.fitness,

        )

        probability = np.arange(

            len(ranked),

            0,

            -1,

        )

        probability = (

            probability

            / probability.sum()

        )

        index = np.random.choice(

            len(ranked),

            p=probability,

        )

        return ranked[index].copy()