"""
=========================================================
Dataset Loader
=========================================================

Supports:

1. NSL-KDD
2. UNSW-NB15
3. CICIDS2017

Author : Viraj Mohod
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger


class DatasetLoader:
    """
    Generic dataset loader.
    """

    def __init__(self, dataset_root: str | Path):

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_root}"
            )

        logger.info(f"Dataset root: {self.dataset_root}")

    def load_csv(
        self,
        filename: str,
        **kwargs
    ) -> pd.DataFrame:

        path = self.dataset_root / filename

        if not path.exists():
            raise FileNotFoundError(path)

        logger.info(f"Loading {path.name}")

        df = pd.read_csv(path, **kwargs)

        logger.success(
            f"{path.name} loaded "
            f"({df.shape[0]} rows × {df.shape[1]} columns)"
        )

        return df

    def load_nsl_kdd(
        self,
        train_file: str,
        test_file: str,
    ) -> Dict[str, pd.DataFrame]:

        logger.info("Loading NSL-KDD")

        train = self.load_csv(train_file, header=None)

        test = self.load_csv(test_file, header=None)

        return {
            "train": train,
            "test": test,
        }

    def load_unsw_nb15(
        self,
        train_file: str,
        test_file: str,
    ) -> Dict[str, pd.DataFrame]:

        logger.info("Loading UNSW-NB15")

        train = self.load_csv(train_file)

        test = self.load_csv(test_file)

        return {
            "train": train,
            "test": test,
        }

    def load_cicids(
        self,
        folder: str,
    ) -> pd.DataFrame:

        folder = self.dataset_root / folder

        if not folder.exists():
            raise FileNotFoundError(folder)

        logger.info("Reading CICIDS2017 CSV files")

        csv_files = sorted(folder.glob("*.csv"))

        if len(csv_files) == 0:
            raise RuntimeError(
                "No CSV files found."
            )

        frames = []

        for csv_file in csv_files:

            logger.info(csv_file.name)

            try:
                df = pd.read_csv(
                    csv_file,
                    low_memory=False,
                )

                frames.append(df)

            except Exception as exc:
                logger.error(exc)

        dataset = pd.concat(
            frames,
            ignore_index=True,
        )

        logger.success(
            f"CICIDS2017 Loaded: {dataset.shape}"
        )

        return dataset

    @staticmethod
    def dataset_info(df: pd.DataFrame):

        print("=" * 60)

        print("Rows :", df.shape[0])

        print("Columns :", df.shape[1])

        print()

        print(df.dtypes)

        print("=" * 60)

    @staticmethod
    def memory_usage(df: pd.DataFrame):

        memory = (
            df.memory_usage(deep=True)
            .sum()
            / (1024 ** 2)
        )

        print(
            f"Memory Usage : {memory:.2f} MB"
        )

    @staticmethod
    def missing_values(df: pd.DataFrame):

        missing = (
            df.isna()
            .sum()
            .sort_values(ascending=False)
        )

        return missing[missing > 0]

    @staticmethod
    def duplicate_rows(df: pd.DataFrame):

        return int(df.duplicated().sum())

    @staticmethod
    def class_distribution(
        df: pd.DataFrame,
        target: str,
    ):

        return (
            df[target]
            .value_counts()
            .sort_values(
                ascending=False
            )
        )

    @staticmethod
    def numeric_columns(df):

        return list(
            df.select_dtypes(
                include="number"
            ).columns
        )

    @staticmethod
    def categorical_columns(df):

        return list(
            df.select_dtypes(
                exclude="number"
            ).columns
        )