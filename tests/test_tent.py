from algorithms.proposed.tent_map import TentChaoticMap

tent = TentChaoticMap(

    population_size=10,

    dimension=20,

)

population = tent.initialize()

print(population)

print()

print(population.shape)