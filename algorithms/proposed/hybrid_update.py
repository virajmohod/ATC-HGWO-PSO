"""
=========================================================
Adaptive Hybrid GWO-PSO Update Engine
=========================================================

Core optimization engine for ATC-HGWO-PSO

Features
--------
✓ Adaptive GWO guidance
✓ Adaptive PSO velocity
✓ Elite archive guidance
✓ Diversity-aware exploration
✓ Binary feature selection
✓ Dynamic exploration/exploitation

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np

from algorithms.proposed.adaptive_gwo import AdaptiveGWO
from algorithms.proposed.adaptive_pso import AdaptivePSO


class HybridUpdate:

    def __init__(

        self,

        dimension,

        population_size,

    ):

        self.dimension = dimension

        self.population_size = population_size

        self.gwo = AdaptiveGWO(

            dimension

        )

        self.pso = AdaptivePSO(

            dimension,

            population_size,

        )

    # ---------------------------------------------------------

    @staticmethod
    def sigmoid(x):

        return 1.0 / (

            1.0 +

            np.exp(-x)

        )

    # ---------------------------------------------------------

    @staticmethod
    def repair(mask):

        if np.sum(mask) == 0:

            mask[

                np.random.randint(

                    len(mask)

                )

            ] = 1

        return mask
    
    # ---------------------------------------------------------

    def hybrid_weight(

        self,

        iteration,

        max_iteration,

    ):

        """
        Adaptive hybrid weight.

        Early iterations

            GWO dominates.

        Later iterations

            PSO dominates.
        """

        progress = (

            iteration

            /

            max_iteration

        )

        gwo_weight = (

            1

            -

            progress

        )

        pso_weight = progress

        return (

            gwo_weight,

            pso_weight,

        )
    # ---------------------------------------------------------

    def elite_guidance(

        self,

        archive,

    ):

        """
        Choose one elite solution.

        Random elite improves diversity.
        """

        elite = archive.random()

        if elite is None:

            return None

        return elite.position
    # ---------------------------------------------------------

    def combine(

        self,

        gwo_solution,

        pso_solution,

        gwo_weight,

        pso_weight,

    ):

        combined = (

            gwo_weight

            *

            gwo_solution

            +

            pso_weight

            *

            pso_solution

        )

        probability = self.sigmoid(

            combined

        )

        binary = (

            np.random.rand(

                self.dimension

            )

            < probability

        ).astype(int)

        return self.repair(

            binary

        )
    # ---------------------------------------------------------

    def update(

        self,

        particle,

        index,

        alpha,

        beta,

        delta,

        personal_best,

        global_best,

        elite_archive,

        iteration,

        max_iteration,

    ):

        gwo_weight,

        pso_weight = self.hybrid_weight(

            iteration,

            max_iteration,

        )

        gwo_candidate = self.gwo.update_position(

            particle,

            alpha,

            beta,

            delta,

            iteration,

            max_iteration,

        )
        pso_candidate = self.pso.update_particle(

            particle,

            personal_best,

            global_best,

            index,

            iteration,

            max_iteration,

        )
        elite = self.elite_guidance(

            elite_archive

        )

        if elite is not None:

            mask = (

                np.random.rand(

                    self.dimension

                )

                < 0.25

            )

            gwo_candidate[mask] = elite[mask]
        new_particle = self.combine(

            gwo_candidate,

            pso_candidate,

            gwo_weight,

            pso_weight,

        )

        return new_particle
    
