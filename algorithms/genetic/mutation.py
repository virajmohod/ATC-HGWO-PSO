"""
=========================================================
Mutation Operators
=========================================================

Implements

1. Bit Flip Mutation

2. Adaptive Mutation

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome


class Mutation:

    @staticmethod
    def bit_flip(
        chromosome: Chromosome,
        probability: float = 0.02,
    ):

        genes = chromosome.genes.copy()

        for i in range(len(genes)):

            if np.random.rand() < probability:

                genes[i] = 1 - genes[i]

        if genes.sum() == 0:

            genes[np.random.randint(len(genes))] = 1

        chromosome.genes = genes

        return chromosome

    @staticmethod
    def adaptive(
        chromosome: Chromosome,
        iteration: int,
        max_iteration: int,
    ):

        probability = (
            0.30
            * (1 - iteration / max_iteration)
            + 0.01
        )

        return Mutation.bit_flip(
            chromosome,
            probability,
        )