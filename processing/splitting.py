"""
=========================================================
Dataset Splitting Module
=========================================================

Supports:

1. Train-Test Split
2. Stratified Split
3. K-Fold
4. Stratified K-Fold

Author : Viraj Mohod
"""

from __future__ import annotations

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
)

from loguru import logger


class DataSplitter:

    @staticmethod
    def split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=True,
    ):

        logger.info("Creating Train/Test Split")

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

    @staticmethod
    def kfold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    ):

        logger.info("Creating K-Fold")

        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    @staticmethod
    def stratified_kfold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    ):

        logger.info("Creating Stratified K-Fold")

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )