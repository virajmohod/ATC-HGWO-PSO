import numpy as np

from algorithms.proposed import (

    HybridUpdate,

    EliteArchive,

)

dimension = 20

population = 10

hybrid = HybridUpdate(

    dimension,

    population,

)

archive = EliteArchive()

mask = np.random.randint(

    0,

    2,

    dimension,

)

archive.add(

    mask,

    0.12,

)

particle = np.random.randint(

    0,

    2,

    dimension,

)

alpha = np.random.randint(

    0,

    2,

    dimension,

)

beta = np.random.randint(

    0,

    2,

    dimension,

)

delta = np.random.randint(

    0,

    2,

    dimension,

)

updated = hybrid.update(

    particle,

    index=0,

    alpha=alpha,

    beta=beta,

    delta=delta,

    personal_best=particle,

    global_best=alpha,

    elite_archive=archive,

    iteration=20,

    max_iteration=100,

)

print(updated)