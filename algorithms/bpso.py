"""
=========================================================
Binary Particle Swarm Optimization (BPSO)
=========================================================

Implements wrapper-based Binary PSO for feature selection.

Velocity update:
    v = w*v
        + c1*r1*(pbest-x)
        + c2*r2*(gbest-x)

Binary update:
    s(v)=1/(1+exp(-v))

Author : Viraj Mohod
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loguru import logger

from algorithms.base_optimizer import BaseOptimizer


class BinaryPSO(BaseOptimizer):

    def __init__(
        self,
        population_size: int = 30,
        iterations: int = 100,
        alpha: float = 0.99,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        random_state: int = 42,
    ):

        super().__init__(
            population_size=population_size,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
        )

        self.w_max = w_max
        self.w_min = w_min

        self.c1 = c1
        self.c2 = c2

        self.velocity = None
        self.personal_best = None
        self.personal_best_score = None

    # -----------------------------------------------------

    def initialize(self, dimension):

        super().initialize(dimension)

        self.velocity = np.random.uniform(
            -1,
            1,
            (self.population_size, dimension),
        )

        self.personal_best = self.population.copy()

        self.personal_best_score = np.full(
            self.population_size,
            np.inf,
        )

    # -----------------------------------------------------

    def inertia(self, iteration):

        return self.w_max - (
            (self.w_max - self.w_min)
            * iteration
            / self.iterations
        )

    # -----------------------------------------------------

    def evaluate_particles(self, X, y):

        for i in range(self.population_size):

            result = self.wrapper.evaluate(
                self.population[i],
                X,
                y,
            )

            if result.fitness < self.personal_best_score[i]:

                self.personal_best_score[i] = result.fitness

                self.personal_best[i] = (
                    self.population[i].copy()
                )

            if (
                self.best_result is None
                or result.fitness
                < self.best_result.fitness
            ):

                self.best_result = result

                self.best_mask = (
                    self.population[i].copy()
                )

    # -----------------------------------------------------

    def update_particles(self, iteration):

        w = self.inertia(iteration)

        for i in range(self.population_size):

            r1 = np.random.rand(self.dimension)
            r2 = np.random.rand(self.dimension)

            cognitive = (
                self.c1
                * r1
                * (
                    self.personal_best[i]
                    - self.population[i]
                )
            )

            social = (
                self.c2
                * r2
                * (
                    self.best_mask
                    - self.population[i]
                )
            )

            self.velocity[i] = (
                w * self.velocity[i]
                + cognitive
                + social
            )

            probability = self.sigmoid(
                self.velocity[i]
            )

            self.population[i] = (
                np.random.rand(self.dimension)
                < probability
            ).astype(int)

            self.population[i] = self.repair(
                self.population[i]
            )

    # -----------------------------------------------------

    def optimize(
        self,
        X: pd.DataFrame,
        y,
    ):

        logger.info("Running Binary PSO")

        self.initialize(X.shape[1])

        for iteration in range(self.iterations):

            self.evaluate_particles(X, y)

            self.history.append(
                self.best_result.fitness
            )

            logger.info(
                f"Iteration "
                f"{iteration+1}/{self.iterations} "
                f"Fitness="
                f"{self.best_result.fitness:.6f}"
            )

            self.update_particles(iteration)

        logger.success("Binary PSO Completed")

        return self.best()