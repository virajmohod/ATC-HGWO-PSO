"""
=========================================================
Data Cleaning Module
=========================================================

Performs:

1. Missing Value Handling
2. Duplicate Removal
3. Infinite Value Removal
4. Constant Column Removal
5. Optional Outlier Clipping

Author : Viraj Mohod
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loguru import logger


class DataCleaner:

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

        duplicates = df.duplicated().sum()

        logger.info(f"Duplicate Rows Found : {duplicates}")

        df = df.drop_duplicates()

        logger.success(
            f"Dataset Shape After Duplicate Removal : {df.shape}"
        )

        return df

    @staticmethod
    def replace_infinite(df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Replacing Infinite Values")

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    @staticmethod
    def remove_constant_columns(df: pd.DataFrame):

        constant_columns = [
            col
            for col in df.columns
            if df[col].nunique() <= 1
        ]

        logger.info(
            f"Constant Columns Removed : {len(constant_columns)}"
        )

        df = df.drop(columns=constant_columns)

        return df

    @staticmethod
    def missing_value_report(df: pd.DataFrame):

        report = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        return report[report > 0]

    @staticmethod
    def fill_numeric(df: pd.DataFrame):

        numeric = df.select_dtypes(include=np.number).columns

        for col in numeric:

            df[col] = df[col].fillna(df[col].median())

        return df

    @staticmethod
    def fill_categorical(df: pd.DataFrame):

        categorical = df.select_dtypes(exclude=np.number).columns

        for col in categorical:

            mode = df[col].mode()

            if len(mode):

                df[col] = df[col].fillna(mode[0])

        return df

    @staticmethod
    def clip_outliers(df: pd.DataFrame):

        numeric = df.select_dtypes(include=np.number).columns

        for col in numeric:

            q1 = df[col].quantile(0.25)

            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            df[col] = df[col].clip(lower, upper)

        logger.success("Outliers Clipped")

        return df

    @staticmethod
    def clean(df: pd.DataFrame):

        logger.info("Starting Cleaning Pipeline")

        df = DataCleaner.replace_infinite(df)

        df = DataCleaner.remove_duplicates(df)

        df = DataCleaner.remove_constant_columns(df)

        df = DataCleaner.fill_numeric(df)

        df = DataCleaner.fill_categorical(df)

        logger.success("Cleaning Completed")

        return df