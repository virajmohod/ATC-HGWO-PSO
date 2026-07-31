"""
=========================================================
Wrapper Fitness Function
=========================================================
"""

from __future__ import annotations

import numpy as np

from sklearn.model_selection import cross_val_score


class WrapperFitness:

    def __init__(

        self,

        classifier,

        alpha=0.99,

        cv=5,

    ):

        self.classifier = classifier

        self.alpha = alpha

        self.cv = cv

    def evaluate(

        self,

        mask,

        X,

        y,

    ):

        if np.sum(mask) == 0:

            mask[np.random.randint(len(mask))] = 1

        selected = np.where(mask == 1)[0]

        X_selected = X.iloc[:, selected]

        accuracy = cross_val_score(

            self.classifier,

            X_selected,

            y,

            cv=self.cv,

            scoring="accuracy",

        ).mean()

        feature_ratio = len(selected) / X.shape[1]

        fitness = (

            self.alpha * (1 - accuracy)

            +

            (1 - self.alpha) * feature_ratio

        )

        from models.result import OptimizationResult

        return OptimizationResult(

            fitness=fitness,

            accuracy=accuracy,

            precision=0,

            recall=0,

            f1_score=0,

            selected_features=selected.tolist(),

            feature_mask=mask.copy(),

        )