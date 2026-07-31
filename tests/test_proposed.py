import pandas as pd

from sklearn.datasets import load_breast_cancer

from algorithms.proposed.atc_hgwo_pso import ATCHGWOPSO

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

optimizer = ATCHGWOPSO(

    population_size=20,

    iterations=50,

)

result = optimizer.optimize(

    X,

    y,

)

optimizer.summary()

print()

print(result)

print()

print(result.selected_features)