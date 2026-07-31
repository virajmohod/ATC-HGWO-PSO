import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.woa import WhaleOptimizationAlgorithm

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = WhaleOptimizationAlgorithm(

    population_size=20,

    iterations=30,

)

result = optimizer.optimize(

    X,

    y,

)

print()

print("Best Fitness :", result.best_fitness)

print("Accuracy :", result.best_accuracy)

print("Selected Features")

print(result.selected_features)

print("No. Features :", len(result.selected_features))