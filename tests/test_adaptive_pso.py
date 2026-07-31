import numpy as np

from algorithms.proposed import AdaptivePSO

dimension = 20

population = 10

optimizer = AdaptivePSO(

    dimension,

    population,

)

particle = np.random.randint(

    0,

    2,

    dimension,

)

personal = particle.copy()

global_best = np.random.randint(

    0,

    2,

    dimension,

)

new_particle = optimizer.update_particle(

    particle,

    personal,

    global_best,

    index=0,

    iteration=5,

    max_iteration=100,

)

print(new_particle)