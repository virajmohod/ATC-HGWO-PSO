"""
=========================================================
Crossover Operators
=========================================================

Implements:
1. Single Point Crossover
2. Two Point Crossover
3. Uniform Crossover

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome


class Crossover:

    @staticmethod
    def single_point(parent1: Chromosome,
                     parent2: Chromosome):

        length = len(parent1)

        point = np.random.randint(1, length - 1)

        child1 = np.concatenate([
            parent1.genes[:point],
            parent2.genes[point:]
        ])

        child2 = np.concatenate([
            parent2.genes[:point],
            parent1.genes[point:]
        ])

        return (
            Chromosome(child1),
            Chromosome(child2)
        )

    @staticmethod
    def two_point(parent1: Chromosome,
                  parent2: Chromosome):

        length = len(parent1)

        p1, p2 = sorted(
            np.random.choice(
                range(1, length - 1),
                2,
                replace=False
            )
        )

        child1 = parent1.genes.copy()
        child2 = parent2.genes.copy()

        child1[p1:p2] = parent2.genes[p1:p2]
        child2[p1:p2] = parent1.genes[p1:p2]

        return (
            Chromosome(child1),
            Chromosome(child2)
        )

    @staticmethod
    def uniform(parent1: Chromosome,
                parent2: Chromosome):

        mask = np.random.randint(
            0,
            2,
            len(parent1)
        ).astype(bool)

        child1 = parent1.genes.copy()
        child2 = parent2.genes.copy()

        child1[mask] = parent2.genes[mask]
        child2[mask] = parent1.genes[mask]

        return (
            Chromosome(child1),
            Chromosome(child2)
        )