"""
=========================================================
ATC-HGWO-PSO

Adaptive Tent Chaotic
Hybrid Grey Wolf Optimizer
Particle Swarm Optimization

Author : Viraj Mohod

=========================================================
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from loguru import logger

from algorithms.base_optimizer import BaseOptimizer

from algorithms.proposed import (

    TentChaoticMap,

    AdaptivePSO,

    AdaptiveGWO,

    HybridUpdate,

    EliteArchive,

    DiversityController,

)


class ATCHGWOPSO(BaseOptimizer):

    """
    Proposed Feature Selection Algorithm

    Components
    ----------

    ✓ Tent Chaotic Initialization

    ✓ Adaptive GWO

    ✓ Adaptive PSO

    ✓ Elite Archive

    ✓ Diversity Controller

    ✓ Hybrid Update

    """

    def __init__(

        self,

        population_size=30,

        iterations=100,

        alpha=0.99,

        archive_size=10,

        random_state=42,

    ):

        super().__init__(

            population_size,

            iterations,

            alpha,

            random_state,

        )

        self.archive = EliteArchive(

            archive_size

        )

        self.diversity = DiversityController()

        self.hybrid = None

        self.personal_best = None

        self.personal_best_fitness = None

        self.start_time = None
    # ------------------------------------------------------

    def initialize(

        self,

        dimension,

    ):

        self.dimension = dimension

        initializer = TentChaoticMap(

            self.population_size,

            dimension,

        )

        self.population = initializer.initialize()

        self.hybrid = HybridUpdate(

            dimension,

            self.population_size,

        )

        self.personal_best = self.population.copy()

        self.personal_best_fitness = np.full(

            self.population_size,

            np.inf,

        )

        logger.info(

            "Tent chaotic initialization completed."

        )
    # ------------------------------------------------------

    def evaluate_population(

        self,

        X,

        y,

    ):

        for i in range(

            self.population_size

        ):

            result = self.wrapper.evaluate(

                self.population[i],

                X,

                y,

            )

            if (

                result.fitness

                <

                self.personal_best_fitness[i]

            ):

                self.personal_best_fitness[i] = (

                    result.fitness

                )

                self.personal_best[i] = (

                    self.population[i].copy()

                )

            self.archive.add(

                self.population[i],

                result.fitness,

            )

            if (

                self.best_result is None

                or

                result.fitness

                <

                self.best_result.fitness

            ):

                self.best_result = result

                self.best_mask = (

                    self.population[i].copy()

                )
    # ------------------------------------------------------

    def leaders(self):

        elites = self.archive.top(3)

        if len(elites) < 3:

            return (

                self.best_mask,

                self.best_mask,

                self.best_mask,

            )

        return (

            elites[0].position,

            elites[1].position,

            elites[2].position,

        )
    # ------------------------------------------------------

    def update_population(

        self,

        iteration,

    ):

        alpha,

        beta,

        delta = self.leaders()

        global_best = self.best_mask.copy()

        new_population = []
        for i in range(

            self.population_size

        ):

            particle = self.hybrid.update(

                particle=self.population[i],

                index=i,

                alpha=alpha,

                beta=beta,

                delta=delta,

                personal_best=self.personal_best[i],

                global_best=global_best,

                elite_archive=self.archive,

                iteration=iteration,

                max_iteration=self.iterations,

            )

            new_population.append(

                particle

            )

        self.population = np.array(

            new_population

        )
        self.diversity.update(

            self.best_result.fitness

        )

        if (

            self.diversity.needs_diversification(

                self.population

            )

            or

            self.diversity.stagnated()

        ):

            logger.info(

                "Population diversified."

            )

            self.population = (

                self.diversity.diversify(

                    self.population,

                    self.archive,

                )

            )
    # ------------------------------------------------------

    def optimize(

        self,

        X: pd.DataFrame,

        y,

    ):

        logger.info(

            "=" * 60

        )

        logger.info(

            "Starting ATC-HGWO-PSO"

        )

        logger.info(

            "=" * 60

        )

        self.start_time = time.time()

        self.initialize(

            X.shape[1]

        )

        best = np.inf

        stagnant = 0

        early_stop = 25

        for iteration in range(

            self.iterations

        ):

            logger.info(

                f"Iteration "

                f"{iteration+1}"

                f"/"

                f"{self.iterations}"

            )

            self.evaluate_population(

                X,

                y,

            )

            self.history.append(

                self.best_result.fitness

            )

            logger.info(

                f"Fitness : "

                f"{self.best_result.fitness:.6f}"

            )

            logger.info(

                f"Accuracy : "

                f"{self.best_result.accuracy:.4f}"

            )

            logger.info(

                f"Features : "

                f"{len(self.best_result.selected_features)}"

            )

            if (

                self.best_result.fitness

                <

                best

            ):

                best = self.best_result.fitness

                stagnant = 0

            else:

                stagnant += 1

            if stagnant >= early_stop:

                logger.success(

                    "Early stopping activated."

                )

                break

            self.update_population(

                iteration

            )

        return self.finalize()
    # ------------------------------------------------------

    def finalize(self):

        execution = (

            time.time()

            -

            self.start_time

        )

        self.best_result.execution_time = (

            execution

        )

        self.best_result.optimizer_name = (

            "ATC-HGWO-PSO"

        )

        logger.success(

            "=" * 60

        )

        logger.success(

            "Optimization Completed"

        )

        logger.success(

            f"Best Fitness : "

            f"{self.best_result.fitness:.6f}"

        )

        logger.success(

            f"Accuracy : "

            f"{self.best_result.accuracy:.4f}"

        )

        logger.success(

            f"Execution : "

            f"{execution:.2f} sec"

        )

        logger.success(

            "=" * 60

        )

        return self.best_result
    # ------------------------------------------------------

    def archive_statistics(

        self,

    ):

        stats = self.archive.statistics()

        if stats is None:

            return None

        logger.info(

            "Elite Archive"

        )

        logger.info(

            f"Solutions : "

            f"{stats['count']}"

        )

        logger.info(

            f"Best : "

            f"{stats['best']:.6f}"

        )

        logger.info(

            f"Mean : "

            f"{stats['mean']:.6f}"

        )

        logger.info(

            f"Std : "

            f"{stats['std']:.6f}"

        )

        return stats
    # ------------------------------------------------------

    def diversity_statistics(

        self,

    ):

        stats = self.diversity.statistics(

            self.population

        )

        logger.info(

            "Population Diversity"

        )

        logger.info(

            stats

        )

        return stats
    # ------------------------------------------------------

    def convergence_curve(

        self,

    ):

        return np.array(

            self.history

        )
    # ------------------------------------------------------

    def selected_mask(

        self,

    ):

        return self.best_mask.copy()
    # ------------------------------------------------------

    def selected_features(

        self,

    ):

        return np.where(

            self.best_mask == 1

        )[0]
    # ------------------------------------------------------

    def summary(

        self,

    ):

        print()

        print("=" * 60)

        print("ATC-HGWO-PSO Summary")

        print("=" * 60)

        print(

            "Fitness :",

            self.best_result.fitness,

        )

        print(

            "Accuracy :",

            self.best_result.accuracy,

        )

        print(

            "Features :",

            len(

                self.best_result.selected_features

            ),

        )

        print(

            "Execution :",

            self.best_result.execution_time,

        )

        print("=" * 60)
# ------------------------------------------------------

def export_convergence(
    self,
    filename="results/convergence.csv",
):

    import os
    import pandas as pd

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    df = pd.DataFrame({

        "Iteration": np.arange(
            1,
            len(self.history) + 1,
        ),

        "Fitness": self.history,

    })

    df.to_csv(
        filename,
        index=False,
    )

    logger.success(

        f"Convergence exported -> {filename}"

    )
# ------------------------------------------------------

def save_feature_mask(
    self,
    filename="results/selected_features.npy",
):

    import os

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    np.save(
        filename,
        self.best_mask,
    )

    logger.success(

        f"Feature mask saved -> {filename}"

    )
# ------------------------------------------------------

def export_metadata(
    self,
    filename="results/metadata.json",
):

    import json
    import os

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    metadata = {

        "optimizer": "ATC-HGWO-PSO",

        "population": self.population_size,

        "iterations": self.iterations,

        "fitness": float(
            self.best_result.fitness
        ),

        "accuracy": float(
            self.best_result.accuracy
        ),

        "selected_features": len(
            self.best_result.selected_features
        ),

        "execution_time":
            self.best_result.execution_time,

    }

    with open(
        filename,
        "w",
    ) as f:

        json.dump(

            metadata,

            f,

            indent=4,

        )

    logger.success(

        f"Metadata exported -> {filename}"

    )
# ------------------------------------------------------

def reset(self):

    self.population = None

    self.history = []

    self.best_mask = None

    self.best_result = None

    self.personal_best = None

    self.personal_best_fitness = None

    self.archive.clear()

    self.diversity = DiversityController()

    logger.info(

        "Optimizer reset."

    )
# ------------------------------------------------------

def run_multiple(
    self,
    X,
    y,
    runs=30,
):

    import pandas as pd

    results = []

    for run in range(runs):

        logger.info(

            f"Run {run+1}/{runs}"

        )

        self.reset()

        result = self.optimize(
            X,
            y,
        )

        results.append({

            "Run": run + 1,

            "Fitness": result.fitness,

            "Accuracy": result.accuracy,

            "Features":
                len(result.selected_features),

            "Time":
                result.execution_time,

        })

    return pd.DataFrame(
        results
    )
# ------------------------------------------------------

def save_runs(
    self,
    dataframe,
    filename="results/runs.csv",
):

    import os

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    dataframe.to_csv(
        filename,
        index=False,
    )
optimizer = ATCHGWOPSO(

    population_size=30,

    iterations=100,

)

result = optimizer.optimize(

    X,

    y,

)

optimizer.export_convergence()

optimizer.export_metadata()

optimizer.save_feature_mask()

runs = optimizer.run_multiple(

    X,

    y,

    runs=30,

)

optimizer.save_runs(runs)

optimizer.summary()
