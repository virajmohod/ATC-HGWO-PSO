from algorithms import (

    BinaryPSO,

    GreyWolfOptimizer,

    GeneticAlgorithm,

    WhaleOptimizationAlgorithm,

    HarrisHawksOptimizer,

    SalpSwarmAlgorithm,

)

from algorithms.proposed.atc_hgwo_pso import (

    ATCHGWOPSO

)


class Benchmark:

    @staticmethod

    def algorithms():

        return {

            "BPSO":BinaryPSO,

            "BGWO":GreyWolfOptimizer,

            "BGA":GeneticAlgorithm,

            "BWOA":WhaleOptimizationAlgorithm,

            "BHHO":HarrisHawksOptimizer,

            "BSSA":SalpSwarmAlgorithm,

            "ATC-HGWO-PSO":ATCHGWOPSO,

        }