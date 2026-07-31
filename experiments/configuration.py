"""
=========================================================
Experiment Configuration
=========================================================
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:

    RANDOM_STATE = 42

    RUNS = 30

    CV = 5

    POPULATION = 30

    ITERATIONS = 100

    ALPHA = 0.99

    CLASSIFIER = "RandomForest"

    OUTPUT_FOLDER = "results"

    SAVE_CONVERGENCE = True

    SAVE_PLOTS = True

    SAVE_STATISTICS = True