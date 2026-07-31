import numpy as np

from algorithms.proposed import DiversityController

population = np.random.randint(

    0,

    2,

    (20,50),

)

controller = DiversityController(

    threshold=0.25,

)

print(

    controller.statistics(

        population

    )

)

controller.update(0.35)

controller.update(0.35)

controller.update(0.35)

print(

    controller.stagnated()

)

population = controller.diversify(

    population

)

print(

    controller.statistics(

        population

    )

)