from datasets.loader import DatasetLoader

loader = DatasetLoader()

for name, X, y in loader.load_all():

    print("=" * 50)

    print("Dataset :", name)

    print("Samples :", X.shape[0])

    print("Features :", X.shape[1])

    print("Classes :", len(set(y)))

    print()