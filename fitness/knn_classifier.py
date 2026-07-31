"""
=========================================================
KNN Wrapper Classifier
=========================================================

Used by every optimization algorithm.

Computes:

- Accuracy
- Precision
- Recall
- F1 Score
- MCC
- ROC AUC
- Confusion Matrix

Author : Viraj Mohod
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

from loguru import logger


@dataclass
class EvaluationResult:

    accuracy: float

    precision: float

    recall: float

    f1: float

    mcc: float

    auc: float

    confusion: np.ndarray


class KNNClassifier:

    def __init__(

        self,

        neighbours: int = 5,

        cv: int = 10,

        random_state: int = 42,

    ):

        self.neighbours = neighbours

        self.cv = cv

        self.random_state = random_state

    def evaluate(

        self,

        X,

        y,

    ) -> EvaluationResult:

        logger.info("Running KNN Cross Validation")

        model = KNeighborsClassifier(

            n_neighbors=self.neighbours

        )

        cv = StratifiedKFold(

            n_splits=self.cv,

            shuffle=True,

            random_state=self.random_state,

        )

        prediction = cross_val_predict(

            model,

            X,

            y,

            cv=cv,

        )

        accuracy = accuracy_score(

            y,

            prediction,

        )

        precision = precision_score(

            y,

            prediction,

            average="weighted",

            zero_division=0,

        )

        recall = recall_score(

            y,

            prediction,

            average="weighted",

            zero_division=0,

        )

        f1 = f1_score(

            y,

            prediction,

            average="weighted",

            zero_division=0,

        )

        mcc = matthews_corrcoef(

            y,

            prediction,

        )

        confusion = confusion_matrix(

            y,

            prediction,

        )

        try:

            probability = cross_val_predict(

                model,

                X,

                y,

                cv=cv,

                method="predict_proba",

            )

            if probability.shape[1] == 2:

                auc = roc_auc_score(

                    y,

                    probability[:, 1],

                )

            else:

                auc = roc_auc_score(

                    y,

                    probability,

                    multi_class="ovr",

                )

        except Exception:

            auc = 0.0

        logger.success(

            f"Accuracy : {accuracy:.4f}"

        )

        return EvaluationResult(

            accuracy=accuracy,

            precision=precision,

            recall=recall,

            f1=f1,

            mcc=mcc,

            auc=auc,

            confusion=confusion,

        )

    @staticmethod
    def print_report(result: EvaluationResult):

        print()

        print("=" * 60)

        print("Evaluation Results")

        print("=" * 60)

        print(f"Accuracy  : {result.accuracy:.4f}")

        print(f"Precision : {result.precision:.4f}")

        print(f"Recall    : {result.recall:.4f}")

        print(f"F1 Score  : {result.f1:.4f}")

        print(f"MCC       : {result.mcc:.4f}")

        print(f"AUC       : {result.auc:.4f}")

        print("=" * 60)

    @staticmethod
    def to_dict(

        result: EvaluationResult,

    ) -> Dict:

        return {

            "Accuracy": result.accuracy,

            "Precision": result.precision,

            "Recall": result.recall,

            "F1": result.f1,

            "MCC": result.mcc,

            "AUC": result.auc,

        }