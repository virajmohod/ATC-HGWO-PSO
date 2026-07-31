import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.gwo import GreyWolfOptimizer

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = GreyWolfOptimizer(

    population_size=20,

    iterations=20,

)

result = optimizer.optimize(

    X,

    y,

)

print()

print("Best Fitness :", result.best_fitness)

print("Accuracy :", result.best_accuracy)

print("Selected Features :")

print(result.selected_features)

print("Number of Features :", len(result.selected_features))