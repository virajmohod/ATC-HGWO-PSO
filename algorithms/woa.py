"""
=========================================================
Binary Whale Optimization Algorithm (BWOA)
=========================================================

Reference:
Mirjalili & Lewis (2016)

Implements:
• Encircling prey
• Bubble-net attacking
• Search for prey
• Binary transfer function
• Wrapper fitness evaluation
• Early stopping
• Convergence history

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loguru import logger

from algorithms.base_optimizer import BaseOptimizer
from algorithms.whale import (
    Encircling,
    SpiralAttack,
    WhalePosition,
)


class WhaleOptimizationAlgorithm(BaseOptimizer):

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

    # -------------------------------------------------

    def evaluate_whales(self, X, y):

        for whale in self.population:

            result = self.wrapper.evaluate(
                whale,
                X,
                y,
            )

            if (
                self.best_result is None
                or result.fitness < self.best_result.fitness
            ):

                self.best_result = result
                self.best_mask = whale.copy()

    # -------------------------------------------------

    def update_population(self, iteration):

        a = 2 - 2 * iteration / self.iterations

        for i in range(self.population_size):

            r = np.random.rand()

            A = 2 * a * np.random.rand() - a

            if r < 0.5:

                if abs(A) < 1:

                    new_position = Encircling.update(

                        self.population[i],

                        self.best_mask,

                        iteration,

                        self.iterations,

                    )

                else:

                    random_whale = self.population[
                        np.random.randint(self.population_size)
                    ]

                    new_position = Encircling.update(

                        self.population[i],

                        random_whale,

                        iteration,

                        self.iterations,

                    )

            else:

                new_position = SpiralAttack.update(

                    self.population[i],

                    self.best_mask,

                )

            binary = WhalePosition.binary(

                new_position

            )

            binary = WhalePosition.repair(binary)

            self.population[i] = binary

    # -------------------------------------------------

    def optimize(
        self,
        X: pd.DataFrame,
        y,
    ):

        logger.info(
            "Starting Whale Optimization Algorithm"
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

            self.evaluate_whales(

                X,

                y,

            )

            self.history.append(

                self.best_result.fitness

            )

            logger.info(

                f"Iteration "

                f"{iteration+1}/{self.iterations}"

                f" Fitness={self.best_result.fitness:.6f}"

            )

            if self.best_result.fitness < best:

                best = self.best_result.fitness

                stagnant = 0

            else:

                stagnant += 1

            if stagnant >= early_stop:

                logger.info(

                    "Early stopping."

                )

                break

            self.update_population(

                iteration

            )

        logger.success(

            "Whale Optimization Finished"

        )

        return self.best()