import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.base_optimizer import BaseOptimizer


class DummyOptimizer(BaseOptimizer):

    def optimize(self, X, y):

        self.initialize(X.shape[1])

        self.evaluate_population(X, y)

        return self.best()


dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = DummyOptimizer(
    population_size=20,
    iterations=5,
)

result = optimizer.optimize(X, y)

print(result.best_fitness)

print(result.best_accuracy)

print(result.selected_features)