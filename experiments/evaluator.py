import time


class Evaluator:

    @staticmethod

    def evaluate(

        optimizer,

        X,

        y,

    ):

        start = time.time()

        result = optimizer.optimize(

            X,

            y,

        )

        result.execution_time = (

            time.time()-start

        )

        return result