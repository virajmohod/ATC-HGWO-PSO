import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.ssa import SalpSwarmAlgorithm

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = SalpSwarmAlgorithm(

    population_size=20,

    iterations=30,

)

result = optimizer.optimize(

    X,

    y,

)

print()

print("Best Fitness :", result.fitness)

print("Accuracy :", result.accuracy)

print("Selected Features :")

print(result.selected_features)

print("Number of Features :", len(result.selected_features))