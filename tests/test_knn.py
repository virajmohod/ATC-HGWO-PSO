from sklearn.datasets import load_breast_cancer

from fitness import KNNClassifier

dataset = load_breast_cancer()

X = dataset.data

y = dataset.target

knn = KNNClassifier()

result = knn.evaluate(X, y)

knn.print_report(result)