import pandas as pd


class ExperimentRecorder:

    def __init__(self):

        self.records = []

    def add(

        self,

        optimizer,

        dataset,

        result,

    ):

        self.records.append(

            {

                "Dataset":dataset,

                "Algorithm":optimizer,

                "Fitness":result.fitness,

                "Accuracy":result.accuracy,

                "Features":len(

                    result.selected_features

                ),

                "Execution Time":

                result.execution_time,

            }

        )

    def dataframe(self):

        return pd.DataFrame(

            self.records

        )

    def save(

        self,

        filename,

    ):

        self.dataframe().to_csv(

            filename,

            index=False,

        )