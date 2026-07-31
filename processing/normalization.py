"""
=========================================================
Normalization Module
=========================================================

Supports:

1. Min-Max Scaling
2. Standard Scaling
3. Robust Scaling

Author : Viraj Mohod
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)

from loguru import logger


class Normalizer:

    def __init__(self):

        self.scaler = None

    def minmax(self, df: pd.DataFrame):

        logger.info("Applying Min-Max Scaling")

        self.scaler = MinMaxScaler()

        numeric = df.select_dtypes(include="number").columns

        df[numeric] = self.scaler.fit_transform(df[numeric])

        logger.success("Min-Max Scaling Completed")

        return df

    def standard(self, df: pd.DataFrame):

        logger.info("Applying Standard Scaling")

        self.scaler = StandardScaler()

        numeric = df.select_dtypes(include="number").columns

        df[numeric] = self.scaler.fit_transform(df[numeric])

        logger.success("Standard Scaling Completed")

        return df

    def robust(self, df: pd.DataFrame):

        logger.info("Applying Robust Scaling")

        self.scaler = RobustScaler()

        numeric = df.select_dtypes(include="number").columns

        df[numeric] = self.scaler.fit_transform(df[numeric])

        logger.success("Robust Scaling Completed")

        return df

    def inverse_transform(self, df: pd.DataFrame):

        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted.")

        numeric = df.select_dtypes(include="number").columns

        df[numeric] = self.scaler.inverse_transform(df[numeric])

        return df