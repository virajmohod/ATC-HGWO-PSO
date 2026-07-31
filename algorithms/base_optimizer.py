"""
=========================================================
Base Optimizer
=========================================================
"""

from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from fitness.wrapper import WrapperFitness


class BaseOptimizer:

    def __init__(

        self,

        population_size,

        iterations,

        alpha,

        random_state,

    ):

        np.random.seed(random_state)

        self.population_size = population_size

        self.iterations = iterations

        self.alpha = alpha

        self.random_state = random_state

        self.population = None

        self.dimension = None

        self.history = []

        self.best_mask = None

        self.best_result = None

        self.wrapper = WrapperFitness(

            RandomForestClassifier(

                random_state=random_state,

                n_estimators=100,

            ),

            alpha,

        )

    def initialize(

        self,

        dimension,

    ):

        self.dimension = dimension

        self.population = np.random.randint(

            0,

            2,

            (

                self.population_size,

                dimension,

            ),

        )

        for row in self.population:

            if np.sum(row) == 0:

                row[np.random.randint(dimension)] = 1

    @staticmethod
    def sigmoid(x):

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def repair(mask):

        if np.sum(mask) == 0:

            mask[np.random.randint(len(mask))] = 1

        return mask

    def best(self):

        return self.best_result