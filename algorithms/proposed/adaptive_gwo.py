"""
=========================================================
Adaptive Grey Wolf Optimizer Component
=========================================================

Adaptive GWO used in ATC-HGWO-PSO

Features
--------
• Adaptive nonlinear convergence factor
• Dynamic Alpha-Beta-Delta weighting
• Improved encircling mechanism
• Binary position update
• Exploration-Exploitation balancing

Author : Viraj Mohod
=========================================================
"""

from __future__ import annotations

import numpy as np


class AdaptiveGWO:

    def __init__(self, dimension):

        self.dimension = dimension

    # -----------------------------------------------------

    def adaptive_a(
        self,
        iteration,
        max_iteration,
    ):
        """
        Nonlinear convergence factor.

        Faster exploration during early iterations
        and smoother exploitation later.
        """

        t = iteration / max_iteration

        return 2.0 * (1 - t ** 2)

    # -----------------------------------------------------

    def leadership_weights(
        self,
        iteration,
        max_iteration,
    ):
        """
        Adaptive Alpha/Beta/Delta weights.
        """

        progress = iteration / max_iteration

        w_alpha = 0.60 + 0.20 * progress
        w_beta = 0.25 - 0.10 * progress
        w_delta = 1.0 - w_alpha - w_beta

        return w_alpha, w_beta, w_delta

    # -----------------------------------------------------

    @staticmethod
    def compute_A_C(a):

        r1 = np.random.rand()
        r2 = np.random.rand()

        A = 2 * a * r1 - a
        C = 2 * r2

        return A, C

    # -----------------------------------------------------

    @staticmethod
    def sigmoid(x):

        return 1.0 / (
            1.0 + np.exp(-x)
        )

    # -----------------------------------------------------

    def update_position(
        self,
        wolf,
        alpha,
        beta,
        delta,
        iteration,
        max_iteration,
    ):

        a = self.adaptive_a(
            iteration,
            max_iteration,
        )

        wa, wb, wd = self.leadership_weights(
            iteration,
            max_iteration,
        )

        new_position = np.zeros(self.dimension)

        for j in range(self.dimension):

            A1, C1 = self.compute_A_C(a)
            A2, C2 = self.compute_A_C(a)
            A3, C3 = self.compute_A_C(a)

            D_alpha = abs(
                C1 * alpha[j] - wolf[j]
            )

            D_beta = abs(
                C2 * beta[j] - wolf[j]
            )

            D_delta = abs(
                C3 * delta[j] - wolf[j]
            )

            X1 = alpha[j] - A1 * D_alpha
            X2 = beta[j] - A2 * D_beta
            X3 = delta[j] - A3 * D_delta

            new_position[j] = (
                wa * X1
                + wb * X2
                + wd * X3
            )

        probability = self.sigmoid(
            new_position
        )

        binary = (
            np.random.rand(self.dimension)
            < probability
        ).astype(int)

        if np.sum(binary) == 0:

            binary[
                np.random.randint(
                    self.dimension
                )
            ] = 1

        return binary