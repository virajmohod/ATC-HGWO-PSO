"""
=========================================================
Experiment Runner

Runs all algorithms on all datasets
for multiple independent executions.

Outputs
-------
✓ Results CSV
✓ Convergence CSV
✓ Selected Features
✓ Runtime
✓ Logs

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import os
import time
import traceback

import pandas as pd

from loguru import logger

from experiments.configuration import ExperimentConfig
from experiments.benchmark import Benchmark
from experiments.recorder import ExperimentRecorder

from datasets.loader import DatasetLoader

from visualization.convergence import ConvergencePlot
from visualization.performance import PerformancePlot

from statistics.wilcoxon import WilcoxonTest
from statistics.friedman import FriedmanTest


class ExperimentRunner:

    def __init__(self):

        self.config = ExperimentConfig()

        self.recorder = ExperimentRecorder()

        self.algorithms = Benchmark.algorithms()

        self.datasets = DatasetLoader()

        self.output = self.config.OUTPUT_FOLDER

        os.makedirs(

            self.output,

            exist_ok=True,

        )

    # ----------------------------------------------------

    def run_algorithm(

        self,

        algorithm_name,

        algorithm,

        dataset_name,

        X,

        y,

    ):

        logger.info(

            f"{algorithm_name}"

        )

        optimizer = algorithm(

            population_size=self.config.POPULATION,

            iterations=self.config.ITERATIONS,

            random_state=self.config.RANDOM_STATE,

        )

        start = time.time()

        result = optimizer.optimize(

            X,

            y,

        )

        runtime = time.time() - start

        result.execution_time = runtime

        self.recorder.add(

            algorithm_name,

            dataset_name,

            result,

        )

        if self.config.SAVE_CONVERGENCE:

            optimizer.export_convergence(

                os.path.join(

                    self.output,

                    f"{dataset_name}_{algorithm_name}_convergence.csv",

                )

            )

        optimizer.save_feature_mask(

            os.path.join(

                self.output,

                f"{dataset_name}_{algorithm_name}_features.npy",

            )

        )

        optimizer.export_metadata(

            os.path.join(

                self.output,

                f"{dataset_name}_{algorithm_name}.json",

            )

        )

    # ----------------------------------------------------

    def run_dataset(

        self,

        dataset_name,

        X,

        y,

    ):

        logger.info(

            "=" * 70

        )

        logger.info(

            dataset_name

        )

        logger.info(

            "=" * 70

        )

        for name, algorithm in self.algorithms.items():

            try:

                self.run_algorithm(

                    name,

                    algorithm,

                    dataset_name,

                    X,

                    y,

                )

            except Exception:

                logger.error(

                    traceback.format_exc()

                )

    # ----------------------------------------------------

    def run(self):

        logger.info(

            "Starting Benchmark"

        )

        for dataset_name, X, y in self.datasets.load_all():

            self.run_dataset(

                dataset_name,

                X,

                y,

            )

        results = self.recorder.dataframe()

        results.to_csv(

            os.path.join(

                self.output,

                "benchmark_results.csv",

            ),

            index=False,

        )

        logger.success(

            "Benchmark Completed"

        )

        return results


if __name__ == "__main__":

    runner = ExperimentRunner()

    df = runner.run()

    print(df.head())