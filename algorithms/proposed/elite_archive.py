"""
=========================================================
Elite Archive
=========================================================

Maintains the best feature subsets discovered during
optimization.

Features
--------
• Top-k elite solutions
• Duplicate removal
• Automatic sorting
• Elite retrieval
• Diversity preservation

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EliteSolution:

    fitness: float
    position: np.ndarray

    def copy(self):

        return EliteSolution(

            fitness=self.fitness,

            position=self.position.copy(),

        )


class EliteArchive:

    def __init__(

        self,

        archive_size=10,

    ):

        self.archive_size = archive_size

        self.archive = []

    # ----------------------------------------------------------

    def __len__(self):

        return len(self.archive)

    # ----------------------------------------------------------

    def clear(self):

        self.archive = []

    # ----------------------------------------------------------

    def contains(

        self,

        position,

    ):

        for elite in self.archive:

            if np.array_equal(

                elite.position,

                position,

            ):

                return True

        return False

    # ----------------------------------------------------------

    def add(

        self,

        position,

        fitness,

    ):

        """
        Insert solution if unique.
        """

        if self.contains(position):

            return

        solution = EliteSolution(

            fitness,

            position.copy(),

        )

        self.archive.append(solution)

        self.archive.sort(

            key=lambda x: x.fitness

        )

        if len(self.archive) > self.archive_size:

            self.archive.pop()

    # ----------------------------------------------------------

    def best(self):

        if len(self.archive) == 0:

            return None

        return self.archive[0].copy()

    # ----------------------------------------------------------

    def worst(self):

        if len(self.archive) == 0:

            return None

        return self.archive[-1].copy()

    # ----------------------------------------------------------

    def random(self):

        if len(self.archive) == 0:

            return None

        index = np.random.randint(

            len(self.archive)

        )

        return self.archive[index].copy()

    # ----------------------------------------------------------

    def mean_position(self):

        if len(self.archive) == 0:

            return None

        positions = np.array(

            [

                elite.position

                for elite in self.archive

            ]

        )

        mean = np.mean(

            positions,

            axis=0,

        )

        return (

            mean >= 0.5

        ).astype(int)

    # ----------------------------------------------------------

    def top(self, n=3):

        return [

            elite.copy()

            for elite in self.archive[:n]

        ]

    # ----------------------------------------------------------

    def statistics(self):

        if len(self.archive) == 0:

            return None

        fitness = [

            elite.fitness

            for elite in self.archive

        ]

        return {

            "count": len(fitness),

            "best": np.min(fitness),

            "worst": np.max(fitness),

            "mean": np.mean(fitness),

            "std": np.std(fitness),

        }