"""
=========================================================
Encoding Module
=========================================================

Supports:

1. Label Encoding
2. One Hot Encoding
3. Automatic Encoding

Author : Viraj Mohod
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from loguru import logger


class Encoder:

    def __init__(self):

        self.encoders = {}

    def label_encode(
        self,
        df: pd.DataFrame,
        columns=None
    ):

        if columns is None:

            columns = df.select_dtypes(
                include="object"
            ).columns

        for column in columns:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(
                df[column].astype(str)
            )

            self.encoders[column] = encoder

        logger.success(
            f"Label Encoded : {len(columns)} Columns"
        )

        return df

    @staticmethod
    def one_hot_encode(
        df: pd.DataFrame,
        columns=None
    ):

        if columns is None:

            columns = df.select_dtypes(
                include="object"
            ).columns

        logger.info("Applying One-Hot Encoding")

        df = pd.get_dummies(

            df,

            columns=columns,

            drop_first=True,

            dtype=int

        )

        logger.success("One-Hot Encoding Completed")

        return df

    def transform_target(
        self,
        target: pd.Series
    ):

        encoder = LabelEncoder()

        encoded = encoder.fit_transform(target)

        self.encoders["target"] = encoder

        return encoded

    @staticmethod
    def detect_categorical(df):

        return list(

            df.select_dtypes(

                include="object"

            ).columns

        )

    def inverse_transform(
        self,
        column,
        values
    ):

        return self.encoders[column].inverse_transform(values)