"""
=========================================================
Chromosome Representation
=========================================================

Binary chromosome representation for wrapper feature
selection.

Author : Viraj Mohod
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chromosome:

    genes: np.ndarray

    fitness: float = float("inf")

    accuracy: float = 0.0

    selected_features: int = 0

    def copy(self):

        return Chromosome(

            genes=self.genes.copy(),

            fitness=self.fitness,

            accuracy=self.accuracy,

            selected_features=self.selected_features,

        )

    @staticmethod
    def random(length: int):

        genes = np.random.randint(

            0,

            2,

            length,

        )

        if np.sum(genes) == 0:

            genes[np.random.randint(length)] = 1

        return Chromosome(genes)

    def repair(self):

        if np.sum(self.genes) == 0:

            self.genes[

                np.random.randint(len(self.genes))

            ] = 1

    def feature_indices(self):

        return np.where(

            self.genes == 1

        )[0]

    def __len__(self):

        return len(self.genes)

    def __str__(self):

        return (

            f"Fitness={self.fitness:.6f} | "

            f"Features={self.selected_features}"

        )