"""
=========================================================
Adaptive Particle Swarm Optimization
=========================================================

Adaptive PSO Component used in ATC-HGWO-PSO

Features
--------
• Adaptive inertia weight
• Adaptive cognitive coefficient
• Adaptive social coefficient
• Velocity clamping
• Binary PSO update
• Dynamic parameter scheduling

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np


class AdaptivePSO:

    def __init__(
        self,
        dimension,
        population_size,
        vmax=6.0,
    ):

        self.dimension = dimension
        self.population_size = population_size

        self.vmax = vmax
        self.vmin = -vmax

        self.velocity = np.random.uniform(
            self.vmin,
            self.vmax,
            (
                population_size,
                dimension,
            ),
        )

    # ---------------------------------------------------------

    def inertia_weight(
        self,
        iteration,
        max_iteration,
    ):

        """
        Adaptive inertia weight

        decreases from 0.9 → 0.4
        """

        return (

            0.9

            -

            0.5

            * iteration

            / max_iteration

        )

    # ---------------------------------------------------------

    def cognitive_coefficient(
        self,
        iteration,
        max_iteration,
    ):

        """
        c1 decreases

        2.5 -> 0.5
        """

        return (

            2.5

            -

            2

            * iteration

            / max_iteration

        )

    # ---------------------------------------------------------

    def social_coefficient(
        self,
        iteration,
        max_iteration,
    ):

        """
        c2 increases

        0.5 -> 2.5
        """

        return (

            0.5

            +

            2

            * iteration

            / max_iteration

        )

    # ---------------------------------------------------------

    @staticmethod
    def sigmoid(x):

        return 1.0 / (

            1.0

            +

            np.exp(-x)

        )

    # ---------------------------------------------------------

    def clamp_velocity(self):

        self.velocity = np.clip(

            self.velocity,

            self.vmin,

            self.vmax,

        )

    # ---------------------------------------------------------

    def update_particle(

        self,

        particle,

        pbest,

        gbest,

        index,

        iteration,

        max_iteration,

    ):

        w = self.inertia_weight(

            iteration,

            max_iteration,

        )

        c1 = self.cognitive_coefficient(

            iteration,

            max_iteration,

        )

        c2 = self.social_coefficient(

            iteration,

            max_iteration,

        )

        r1 = np.random.rand(

            self.dimension

        )

        r2 = np.random.rand(

            self.dimension

        )

        self.velocity[index] = (

            w

            * self.velocity[index]

            +

            c1

            * r1

            * (

                pbest

                - particle

            )

            +

            c2

            * r2

            * (

                gbest

                - particle

            )

        )

        self.clamp_velocity()

        probability = self.sigmoid(

            self.velocity[index]

        )

        new_particle = (

            np.random.rand(

                self.dimension

            )

            < probability

        ).astype(int)

        if np.sum(new_particle) == 0:

            new_particle[

                np.random.randint(

                    self.dimension

                )

            ] = 1

        return new_particle