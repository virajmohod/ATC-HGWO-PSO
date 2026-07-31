"""
=========================================================
Binary Salp Swarm Algorithm (BSSA)
=========================================================

Reference:
Mirjalili et al. (2017)

Implements:
• Leader Salp Update
• Follower Chain Update
• Adaptive c1 Parameter
• Binary Transfer Function
• Wrapper Fitness Evaluation
• Early Stopping
• Convergence Tracking

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loguru import logger

from algorithms.base_optimizer import BaseOptimizer
from algorithms.salp import (
    LeaderUpdate,
    FollowerUpdate,
    SalpPosition,
)


class SalpSwarmAlgorithm(BaseOptimizer):

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

    # -------------------------------------------------------

    def evaluate_salps(self, X, y):

        for salp in self.population:

            result = self.wrapper.evaluate(

                salp,

                X,

                y,

            )

            if (

                self.best_result is None

                or result.fitness < self.best_result.fitness

            ):

                self.best_result = result

                self.best_mask = salp.copy()

    # -------------------------------------------------------

    def update_population(self, iteration):

        food = self.best_mask.copy()

        new_population = self.population.copy()

        # Leader update
        new_population[0] = LeaderUpdate.update(

            self.population[0],

            food,

            iteration,

            self.iterations,

        )

        new_population[0] = SalpPosition.binary(

            new_population[0]

        )

        new_population[0] = SalpPosition.repair(

            new_population[0]

        )

        # Follower update
        for i in range(1, self.population_size):

            new_population[i] = FollowerUpdate.update(

                new_population[i - 1],

                self.population[i],

            )

            new_population[i] = SalpPosition.binary(

                new_population[i]

            )

            new_population[i] = SalpPosition.repair(

                new_population[i]

            )

        self.population = new_population

    # -------------------------------------------------------

    def optimize(

        self,

        X: pd.DataFrame,

        y,

    ):

        logger.info(

            "Starting Binary Salp Swarm Algorithm"

        )

        self.initialize(

            X.shape[1]

        )

        best = np.inf

        stagnant = 0

        early_stop = 20

        for iteration in range(

            self.iterations

        ):

            self.evaluate_salps(

                X,

                y,

            )

            self.history.append(

                self.best_result.fitness

            )

            logger.info(

                f"Iteration "

                f"{iteration+1}/{self.iterations} "

                f"Fitness={self.best_result.fitness:.6f}"

            )

            if self.best_result.fitness < best:

                best = self.best_result.fitness

                stagnant = 0

            else:

                stagnant += 1

            if stagnant >= early_stop:

                logger.info(

                    "Early stopping activated."

                )

                break

            self.update_population(

                iteration

            )

        logger.success(

            "Binary Salp Swarm Optimization Finished"

        )

        return self.best_result