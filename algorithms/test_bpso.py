import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.bpso import BinaryPSO

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = BinaryPSO(
    population_size=20,
    iterations=20,
)

result = optimizer.optimize(X, y)

print()

print("Best Fitness :", result.best_fitness)

print("Accuracy :", result.best_accuracy)

print("Selected Features :")

print(result.selected_features)

print("Total :", len(result.selected_features))