from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

import tools
import datasets
import superfastcode


def Main():
    np.random.seed(42)
    #datasets = joblib.load('datasets.pkl')
    #names = ("adult", "dutch_census", "german_credit", "bank_marketing", "law_school")
    ax = plt.gca()
    #for i, name in enumerate(names):
    #    dataset = datasets[name]
    #    X = dataset["X"]
    #    y = dataset["y"]
    #    s = dataset["s"]
    #    tree = superfastcode.FDTC()
    #    tree.fit(X=X.to_numpy(), y=y.to_numpy(), s=s.to_numpy())
    #    pred = tree.predict(X.to_numpy())

    #    RocCurveDisplay.from_predictions(pred, y, ax=ax, alpha=0.7, name=name)

    #plt.show()

    X, y, s = datasets.complex_variable(10000, 10, 0.3)

    for n in range(5):
        print(0)
        tree = superfastcode.FDTC()
        print(1)
        tree.fit(X=X[n], y=y[n], s=s[n])
        print(2)
        pred = tree.predict(X[n])
        print(3)
        RocCurveDisplay.from_predictions(pred, y[n], ax=ax, alpha=0.7, name="generated"+str(n))
    plt.show()
        

if __name__ == "__main__":
    Main()
        