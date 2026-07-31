"""
=========================================================
Optimization Result Model
=========================================================

Common result object returned by all optimization algorithms.

Author : Viraj Mohod
=========================================================
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class OptimizationResult:

    fitness: float

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    selected_features: list

    feature_mask: np.ndarray

    execution_time: float = 0.0

    classifier_name: str = ""

    optimizer_name: str = ""