import numpy as np

from sklearn.datasets import load_breast_cancer

import pandas as pd

from fitness import WrapperFitness

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data)

y = dataset.target

fitness = WrapperFitness()

mask = np.random.randint(

    0,

    2,

    size=X.shape[1]

)

result = fitness.evaluate(

    mask,

    X,

    y,

)

fitness.summary(result)