"""
=========================================================
Binary Harris Hawks Optimization (BHHO)
=========================================================

Reference:
Heidari et al. (2019)

Implements:
• Exploration Phase
• Soft Besiege
• Hard Besiege
• Progressive Rapid Dives
• Levy Flight
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

from algorithms.hawks import (
    Exploration,
    Exploitation,
    HawkPosition,
)


class HarrisHawksOptimizer(BaseOptimizer):

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

    # ----------------------------------------------------------

    def evaluate_hawks(self, X, y):

        for hawk in self.population:

            result = self.wrapper.evaluate(

                hawk,

                X,

                y,

            )

            if (

                self.best_result is None

                or result.fitness < self.best_result.fitness

            ):

                self.best_result = result

                self.best_mask = hawk.copy()

    # ----------------------------------------------------------

    def update_population(

        self,

        iteration,

    ):

        rabbit = self.best_mask.copy()

        mean_position = np.mean(

            self.population,

            axis=0,

        )

        E1 = 2 * (

            1

            - iteration / self.iterations

        )

        for i in range(

            self.population_size

        ):

            E0 = 2 * np.random.rand() - 1

            escaping_energy = E1 * E0

            if abs(escaping_energy) >= 1:

                random_hawk = self.population[

                    np.random.randint(

                        self.population_size

                    )

                ]

                new_position = Exploration.random_search(

                    self.population[i],

                    random_hawk,

                    mean_position,

                    escaping_energy,

                )

            else:

                r = np.random.rand()

                if r >= 0.5:

                    new_position = Exploitation.soft_besiege(

                        self.population[i],

                        rabbit,

                        escaping_energy,

                    )

                else:

                    new_position = Exploitation.rapid_dives(

                        self.population[i],

                        rabbit,

                        escaping_energy,

                    )

            binary = HawkPosition.binary(

                new_position

            )

            binary = HawkPosition.repair(

                binary

            )

            self.population[i] = binary

    # ----------------------------------------------------------

    def optimize(

        self,

        X: pd.DataFrame,

        y,

    ):

        logger.info(

            "Starting Harris Hawks Optimization"

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

            self.evaluate_hawks(

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

            if (

                self.best_result.fitness

                < best

            ):

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

            "Harris Hawks Optimization Finished"

        )

        return self.best()