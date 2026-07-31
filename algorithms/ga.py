"""
=========================================================
Binary Genetic Algorithm for Wrapper Feature Selection
=========================================================

Implements:

• Binary Chromosomes
• Tournament/Roulette/Rank Selection
• Single/Two-point/Uniform Crossover
• Adaptive Mutation
• Elitism
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

from algorithms.genetic import (
    Chromosome,
    Selection,
    Crossover,
    Mutation,
    Elitism,
)


class GeneticAlgorithm(BaseOptimizer):

    def __init__(
        self,
        population_size=30,
        iterations=100,
        alpha=0.99,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elite_size=2,
        selection="tournament",
        crossover="uniform",
        random_state=42,
    ):

        super().__init__(
            population_size,
            iterations,
            alpha,
            random_state,
        )

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.selection_method = selection
        self.crossover_method = crossover

        self.population = []

    # -----------------------------------------------------

    def initialize(self, dimension):

        self.dimension = dimension

        self.population = [
            Chromosome.random(dimension)
            for _ in range(self.population_size)
        ]

    # -----------------------------------------------------

    def evaluate_population(self, X, y):

        for chromosome in self.population:

            result = self.wrapper.evaluate(
                chromosome.genes,
                X,
                y,
            )

            chromosome.fitness = result.fitness
            chromosome.accuracy = result.accuracy
            chromosome.selected_features = len(result.selected_features)

            if (
                self.best_result is None
                or result.fitness < self.best_result.fitness
            ):
                self.best_result = result
                self.best_mask = chromosome.genes.copy()

    # -----------------------------------------------------

    def select_parent(self):

        if self.selection_method == "roulette":

            return Selection.roulette(

                self.population

            )

        elif self.selection_method == "rank":

            return Selection.rank(

                self.population

            )

        return Selection.tournament(

            self.population

        )

    # -----------------------------------------------------

    def crossover(self, p1, p2):

        if np.random.rand() > self.crossover_rate:

            return p1.copy(), p2.copy()

        if self.crossover_method == "single":

            return Crossover.single_point(

                p1,

                p2,

            )

        elif self.crossover_method == "two":

            return Crossover.two_point(

                p1,

                p2,

            )

        return Crossover.uniform(

            p1,

            p2,

        )

    # -----------------------------------------------------

    def mutation(

        self,

        chromosome,

        iteration,

    ):

        chromosome = Mutation.adaptive(

            chromosome,

            iteration,

            self.iterations,

        )

        return chromosome

    # -----------------------------------------------------

    def optimize(

        self,

        X: pd.DataFrame,

        y,

    ):

        logger.info(

            "Starting Genetic Algorithm"

        )

        self.initialize(

            X.shape[1]

        )

        best_fitness = np.inf

        stagnant = 0

        early_stop = 20

        for iteration in range(

            self.iterations

        ):

            self.evaluate_population(

                X,

                y,

            )

            self.population.sort(

                key=lambda c: c.fitness

            )

            self.history.append(
                self.best_result.fitness
            )


            logger.info(

                f"Iteration "

                f"{iteration+1}/{self.iterations} "

                f"Fitness={self.population[0].fitness:.6f}"

            )

            if (

                self.population[0].fitness

                < best_fitness

            ):

                best_fitness = (

                    self.population[0].fitness

                )

                stagnant = 0

            else:

                stagnant += 1

            if stagnant >= early_stop:

                logger.info(

                    "Early stopping activated."

                )

                break

            new_population = Elitism.keep_best(

                self.population,

                self.elite_size,

            )

            while len(new_population) < self.population_size:

                parent1 = self.select_parent()

                parent2 = self.select_parent()

                child1, child2 = self.crossover(

                    parent1,

                    parent2,

                )

                child1 = self.mutation(

                    child1,

                    iteration,

                )

                child2 = self.mutation(

                    child2,

                    iteration,

                )

                child1.repair()

                child2.repair()

                new_population.append(child1)

                if len(new_population) < self.population_size:

                    new_population.append(child2)

            self.population = new_population

        logger.success(

            "Genetic Algorithm Finished"

        )

        return self.best_result