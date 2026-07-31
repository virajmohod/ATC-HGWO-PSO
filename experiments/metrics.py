import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

)


class Metrics:

    @staticmethod
    def evaluate(

        model,

        X_train,

        X_test,

        y_train,

        y_test,

    ):

        model.fit(

            X_train,

            y_train,

        )

        prediction = model.predict(

            X_test

        )

        return {

            "accuracy":

            accuracy_score(

                y_test,

                prediction,

            ),

            "precision":

            precision_score(

                y_test,

                prediction,

                average="weighted",

            ),

            "recall":

            recall_score(

                y_test,

                prediction,

                average="weighted",

            ),

            "f1":

            f1_score(

                y_test,

                prediction,

                average="weighted",

            ),

        }