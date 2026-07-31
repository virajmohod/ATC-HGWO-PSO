"""
=========================================================
Grey Wolf Optimizer (Binary Wrapper Version)
=========================================================

Reference:
Mirjalili et al. (2014)

Author : Viraj Mohod

Description:
Binary Grey Wolf Optimizer for Wrapper Feature Selection
Compatible with BaseOptimizer.

=========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loguru import logger

from algorithms.base_optimizer import BaseOptimizer


class GreyWolfOptimizer(BaseOptimizer):

    def __init__(

        self,

        population_size=30,

        iterations=100,

        alpha=0.99,

        random_state=42,

    ):

        super().__init__(

            population_size,

            iterations,

            alpha,

            random_state,

        )

        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None

        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

    # ------------------------------------------------------

    def evaluate_wolves(self, X, y):

        """
        Evaluate every wolf.
        """

        for i in range(self.population_size):

            result = self.wrapper.evaluate(

                self.population[i],

                X,

                y,

            )

            score = result.fitness

            # Alpha

            if score < self.alpha_score:

                self.delta_score = self.beta_score
                self.delta_wolf = self.beta_wolf

                self.beta_score = self.alpha_score
                self.beta_wolf = self.alpha_wolf

                self.alpha_score = score
                self.alpha_wolf = self.population[i].copy()

                self.best_result = result
                self.best_mask = self.population[i].copy()

            elif score < self.beta_score:

                self.delta_score = self.beta_score
                self.delta_wolf = self.beta_wolf

                self.beta_score = score
                self.beta_wolf = self.population[i].copy()

            elif score < self.delta_score:

                self.delta_score = score
                self.delta_wolf = self.population[i].copy()

    # ------------------------------------------------------

    def update_positions(self, iteration):

        """
        Standard GWO Position Update
        """

        a = 2 - iteration * (2 / self.iterations)

        for i in range(self.population_size):

            for j in range(self.dimension):

                r1 = np.random.rand()
                r2 = np.random.rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(

                    C1 * self.alpha_wolf[j]

                    - self.population[i][j]

                )

                X1 = self.alpha_wolf[j] - A1 * D_alpha

                # ------------------------------

                r1 = np.random.rand()
                r2 = np.random.rand()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(

                    C2 * self.beta_wolf[j]

                    - self.population[i][j]

                )

                X2 = self.beta_wolf[j] - A2 * D_beta

                # ------------------------------

                r1 = np.random.rand()
                r2 = np.random.rand()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(

                    C3 * self.delta_wolf[j]

                    - self.population[i][j]

                )

                X3 = self.delta_wolf[j] - A3 * D_delta

                value = (X1 + X2 + X3) / 3

                probability = self.sigmoid(value)

                self.population[i][j] = (

                    np.random.rand()

                    < probability

                )

            self.population[i] = self.repair(

                self.population[i]

            )

    # ------------------------------------------------------

    def optimize(

        self,

        X: pd.DataFrame,

        y,

    ):

        logger.info(

            "Starting Grey Wolf Optimizer"

        )

        self.initialize(

            X.shape[1]

        )

        for iteration in range(

            self.iterations

        ):

            self.evaluate_wolves(

                X,

                y,

            )

            self.history.append(

                self.alpha_score

            )

            logger.info(

                f"Iteration "

                f"{iteration+1}/"

                f"{self.iterations}"

                f" Fitness={self.alpha_score:.6f}"

            )

            self.update_positions(

                iteration

            )

        logger.success(

            "Grey Wolf Optimization Finished"

        )

        return self.best()