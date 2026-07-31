"""
=========================================================
Dataset Validator
=========================================================
"""

import pandas as pd


class DatasetValidator:

    @staticmethod
    def validate(df):

        if len(df) == 0:
            raise ValueError("Dataset is empty.")

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset requires at least one feature and one target."
            )

        return True