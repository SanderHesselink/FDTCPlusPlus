from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

import tools
import datasets


def Main():
    np.random.seed(42)
    #datasets = joblib.load('datasets.pkl')
    figure, axis = plt.subplots(2,5, figsize=[16, 9])
    #names = ("adult", "dutch_census", "german_credit", "bank_marketing", "law_school")
    #for i, name in enumerate(names):
    #    dataset = datasets[name]
    #    X = dataset["X"]
    #    y = dataset["y"]
    #    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #    categorical_part = X.select_dtypes(exclude=numerics).to_numpy()
    #    for col in range(categorical_part.shape[1]):
    #        categorical_part[:,col] = pd.util.hash_array(vals=categorical_part[:,col])
    #    numeric_part = X.select_dtypes(include=numerics)
    #    X = np.concatenate(
    #        (
    #            numeric_part.values.astype(float),
    #            categorical_part
    #        ), axis=1
    #    )

    #    pca = PCA(n_components=2)

    #    pcax = pca.fit_transform(X)


    #    axis[0, i].scatter(pcax[:,0], pcax[:,1], c=y, cmap="coolwarm", alpha=0.2)
    #    axis[0, i].set_title(name)
    for i, n in [(0, 1000), (1, 10000)]:
        X, y, s = datasets.complex_variable(n, 10, 0.1)

        for j in range(5):
            pca = PCA(n_components=2)
            pcax = pca.fit_transform(X[j])
            print(i, j)
            axis[i, j].scatter(pcax[:,0], pcax[:,1], c=y[j], cmap="coolwarm", alpha=0.2)
            axis[i, j].set_title("generated"+str(j+1))
    plt.show()
        

if __name__ == "__main__":
    Main()
        