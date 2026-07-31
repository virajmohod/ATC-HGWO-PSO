"""
=========================================================
Population Diversity Preservation
=========================================================

Adaptive diversity monitoring for ATC-HGWO-PSO.

Features
--------
• Average Hamming Distance
• Population Diversity Index
• Stagnation Detection
• Adaptive Reinitialization
• Elite Protection
• Diversity Statistics

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np


class DiversityController:

    def __init__(
        self,
        threshold=0.20,
        reinitialize_ratio=0.25,
        stagnation_limit=15,
    ):

        self.threshold = threshold
        self.reinitialize_ratio = reinitialize_ratio
        self.stagnation_limit = stagnation_limit

        self.best_fitness = np.inf
        self.counter = 0

    # ----------------------------------------------------

    @staticmethod
    def hamming_distance(a, b):

        return np.sum(a != b)

    # ----------------------------------------------------

    def average_hamming(self, population):

        n = len(population)

        if n <= 1:
            return 0.0

        total = 0
        pairs = 0

        for i in range(n):

            for j in range(i + 1, n):

                total += self.hamming_distance(
                    population[i],
                    population[j],
                )

                pairs += 1

        return total / pairs

    # ----------------------------------------------------

    def diversity_index(self, population):

        dimension = population.shape[1]

        return self.average_hamming(
            population
        ) / dimension

    # ----------------------------------------------------

    def update(self, current_best):

        if current_best < self.best_fitness:

            self.best_fitness = current_best

            self.counter = 0

        else:

            self.counter += 1

    # ----------------------------------------------------

    def stagnated(self):

        return self.counter >= self.stagnation_limit

    # ----------------------------------------------------

    def needs_diversification(

        self,

        population,

    ):

        diversity = self.diversity_index(

            population

        )

        return diversity < self.threshold

    # ----------------------------------------------------

    def diversify(

        self,

        population,

        elite_archive=None,

    ):

        population = population.copy()

        size = len(population)

        replace = int(

            size * self.reinitialize_ratio

        )

        indices = np.random.choice(

            range(size),

            replace,

            replace=False,

        )

        elite_positions = []

        if elite_archive is not None:

            elite_positions = [

                elite.position

                for elite in elite_archive.archive

            ]

        for idx in indices:

            candidate = np.random.randint(

                0,

                2,

                population.shape[1],

            )

            if candidate.sum() == 0:

                candidate[
                    np.random.randint(
                        population.shape[1]
                    )
                ] = 1

            if len(elite_positions):

                duplicate = any(
                    np.array_equal(
                        candidate,
                        elite,
                    )
                    for elite in elite_positions
                )

                if duplicate:
                    continue

            population[idx] = candidate

        return population

    # ----------------------------------------------------

    def statistics(

        self,

        population,

    ):

        return {

            "diversity": self.diversity_index(
                population
            ),

            "stagnation_counter": self.counter,

            "threshold": self.threshold,

            "reinitialize_ratio":
                self.reinitialize_ratio,

        }