import numpy as np

from algorithms.proposed import AdaptiveGWO

dimension = 20

optimizer = AdaptiveGWO(dimension)

wolf = np.random.randint(0, 2, dimension)
alpha = np.random.randint(0, 2, dimension)
beta = np.random.randint(0, 2, dimension)
delta = np.random.randint(0, 2, dimension)

updated = optimizer.update_position(
    wolf,
    alpha,
    beta,
    delta,
    iteration=10,
    max_iteration=100,
)

print(updated)