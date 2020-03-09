import pandas as pd
import numpy as np


def load_data():
    features = pd.read_excel('CTG.xls', 1)

    labels = np.array(features[23])
    features = features.drop("Unnamed: 44", axis=1)
    features = features.drop(22, axis=1)

    for p in range(31, 43):
        features = features.drop("Unnamed: " + str(p), axis=1)

    for p in range(10):
        features = features.drop("Unnamed: " + str(p), axis=1)

    features = features.drop([0, 2127, 2128, 2129], axis=0)
    features = features.drop(23, axis=1)

    labels = labels[1:-3]
    labels = labels.astype('int')
    features = np.array(features)

    return features, labels
