"""
=========================================================
Elitism Operator
=========================================================

Keeps the best chromosomes from previous generation.

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations


class Elitism:

    @staticmethod
    def keep_best(
        population,
        elite_size=2,
    ):

        population = sorted(

            population,

            key=lambda c: c.fitness

        )

        return [

            chromosome.copy()

            for chromosome in population[:elite_size]

        ]