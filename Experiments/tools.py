import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
import random
import pickle


def make_set(n, n_samples, n_features, s_shift, info_ratio=0.2, redun_ratio=0.2, random_states=None, granularity=100, flip=0.01, sep=1.0, balance=None):
    X = []
    y = []
    s = []

    if random_states == None:
        random_states = list(range(n))

    for random_state in random_states:
        tempX, tempy, temps = make_new(n_samples, n_features, s_shift, info_ratio, redun_ratio, random_state, granularity, flip, sep, balance)
        X.append(tempX)
        y.append(tempy)
        s.append(temps)

    return X, y, s

def make_new(n_samples, n_features, s_shift, info_ratio=0.2, redun_ratio=0.2, random_state=42, granularity=100, flip=0.01, sep=1.0, balance=None):

    random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_informative=round(info_ratio * n_features), 
                               n_redundant=round(redun_ratio * n_features),
                               random_state=random_state,
                               flip_y=flip,
                               class_sep=sep,
                               weights=balance)

    s = []

    for i in y:
        temp = []
        for chance in s_shift:
            value = random.randint(0, granularity) / granularity
            temp.append(str((value <= chance) ^ i))
        s.append(temp)

    s = np.array(s)

    return X, y, s

def save(X, y, s, name):
    name = "Data/Datasets/" + name + ".pkl"
    dictionary = {"X" : X, "y": y, "s" : s}
    with open(name, 'wb') as fp:
        pickle.dump(dictionary, fp)


def load(name):
    name = "Data/Datasets/" + name + ".pkl"
    with open(name, 'rb') as fp:
        dictionary = pickle.load(fp)
    return dictionary["X"], dictionary["y"], dictionary["s"]

def draw(run):
    if not ".pkl" in run:
        run = "Data/Results/" + run + ".pkl"
    with open(run, 'rb') as fp:
        dictionary = pickle.load(fp)
        samples = dictionary["samples"]
        del dictionary["samples"]

        for key in dictionary.keys():
            plt.plot(samples, dictionary[key])
        plt.xlabel("Number of samples")
        plt.ylabel("Runtime in seconds")
        plt.show()