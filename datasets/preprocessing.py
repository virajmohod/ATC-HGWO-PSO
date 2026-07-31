"""
=========================================================
Dataset Preprocessing
=========================================================
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DatasetPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

    def preprocess(self, df):

        df = df.copy()

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Missing value handling
        imputer = SimpleImputer(strategy="mean")

        numeric_columns = df.select_dtypes(include=["number"]).columns

        df[numeric_columns] = imputer.fit_transform(
            df[numeric_columns]
        )

        # Encode categorical features
        for column in df.columns:

            if df[column].dtype == "object":

                encoder = LabelEncoder()

                df[column] = encoder.fit_transform(
                    df[column]
                )

        return df

    def normalize(self, X):

        return self.scaler.fit_transform(X)