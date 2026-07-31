"""
=========================================================
Universal Dataset Loader
=========================================================
"""

from __future__ import annotations

import os
import pandas as pd

from scipy.io import loadmat
from scipy.io import arff

from datasets.preprocessing import DatasetPreprocessor
from datasets.validators import DatasetValidator


class DatasetLoader:

    def __init__(

        self,

        folder="datasets/benchmark",

    ):

        self.folder = folder

        self.preprocessor = DatasetPreprocessor()

    # ---------------------------------------------------

    def load_csv(self, path):

        return pd.read_csv(path)

    # ---------------------------------------------------

    def load_excel(self, path):

        return pd.read_excel(path)

    # ---------------------------------------------------

    def load_arff(self, path):

        data, meta = arff.loadarff(path)

        return pd.DataFrame(data)

    # ---------------------------------------------------

    def load_mat(self, path):

        data = loadmat(path)

        return pd.DataFrame(data["data"])

    # ---------------------------------------------------

    def load_dataset(self, path):

        extension = path.split(".")[-1].lower()

        if extension == "csv":

            df = self.load_csv(path)

        elif extension in ["xls", "xlsx"]:

            df = self.load_excel(path)

        elif extension == "arff":

            df = self.load_arff(path)

        elif extension == "mat":

            df = self.load_mat(path)

        else:

            raise ValueError(
                f"Unsupported dataset: {extension}"
            )

        DatasetValidator.validate(df)

        df = self.preprocessor.preprocess(df)

        X = df.iloc[:, :-1]

        y = df.iloc[:, -1]

        X = self.preprocessor.normalize(X)

        return X, y

    # ---------------------------------------------------

    def load_all(self):

        for file in sorted(os.listdir(self.folder)):

            path = os.path.join(self.folder, file)

            if os.path.isfile(path):

                name = os.path.splitext(file)[0]

                X, y = self.load_dataset(path)

                yield name, X, y