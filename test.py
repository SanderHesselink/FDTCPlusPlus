import warnings
warnings.filterwarnings("ignore")
import os
from random import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import superfastcode
import FDTC
import joblib
import ctpf
import Calc_opt
import OHE_opt
import Experiments.tools as tools

from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_score
from sklearn import tree


n_samples = 10000
n_estimators = 10
# keys: adult, dutch_census, german_credit, bank_marketing, law_school 
dataset = "law_school"
threshold = 0.05


def IAFCTree():
    tree = superfastcode.FDTC()
    t0 = time.perf_counter()
    tree.fit(X=X.to_numpy(), y=y.to_numpy(), s=s.to_numpy())
    t1 = time.perf_counter()

    return tree, str(t1-t0)[:6]+'s'

def IAFPythonTree():
    tree = FDTC.FairDecisionTreeClassifier()
    t0 = time.perf_counter()
    tree.fit(X=X, y=y, s=s)
    t1 = time.perf_counter()

    return tree, str(t1-t0)[:6]+'s'

def IAFCalcoptTree():
    tree = Calc_opt.FairDecisionTreeClassifier()
    t0 = time.perf_counter()
    tree.fit(X=X, y=y, s=s)
    t1 = time.perf_counter()

    return tree, str(t1-t0)[:6]+'s'

def IAFOHEoptTree():
    tree = OHE_opt.FairDecisionTreeClassifier()
    t0 = time.perf_counter()
    tree.fit(X=X, y=y, s=s)
    t1 = time.perf_counter()

    return tree, str(t1-t0)[:6]+'s'

def PPCTree(tree):
    t0 = time.perf_counter()
    pred = tree.predict_proba(X=X.to_numpy())
    t1 = time.perf_counter()

    return np.array(pred)[:,1], str(t1-t0)[:6]+'s'

def PPPythonTree(tree):
    t0 = time.perf_counter()
    pred = tree.predict_proba(X=X)
    t1 = time.perf_counter()

    return pred[:,1], str(t1-t0)[:6]+'s'

def IAFCForest(n_jobs):
    t0 = time.perf_counter()
    forest = superfastcode.FRFC(n_estimators=n_estimators, n_jobs=n_jobs)
    forest.fit(X=X.to_numpy(), y=y.to_numpy(), s=s.to_numpy())
    t1 = time.perf_counter()

    return forest, str(t1-t0)[:6]+'s'

def IAFPythonForest(n_jobs):
    t0 = time.perf_counter()
    forest = FDTC.FairRandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    forest.fit(X=X, y=y, s=s)
    t1 = time.perf_counter()

    return forest, str(t1-t0)[:6]+'s'

def IAFHybridForest(n_jobs):
    t0 = time.perf_counter()
    forest = ctpf.FairRandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    forest.fit(X=X.to_numpy(), y=y.to_numpy(), s=s.to_numpy())
    t1 = time.perf_counter()

    return forest, str(t1-t0)[:6]+'s'

def PPCForest(forest):
    t0 = time.perf_counter()
    pred = forest.predict_proba(X=X.to_numpy())
    t1 = time.perf_counter()

    return np.array(pred)[:,1], str(t1-t0)[:6]+'s'

def PPPythonForest(forest):
    t0 = time.perf_counter()
    pred = forest.predict_proba(X=X)
    t1 = time.perf_counter()

    return pred[:,1], str(t1-t0)[:6]+'s'

def PredDiff(pred1, pred2, threshold = None):
    if threshold == None:
        return (abs(pred1-pred2)).sum() / len(y)
    else:
        return (abs(pred1 - pred2) > threshold).sum() / len(y)

def MSE(pred):
    return ((pred - y)**2).sum() / len(y)


def Main():
    #check_estimator(FDTC.FairDecisionTreeClassifier())
    datasets = joblib.load('datasets.pkl')
    adult = datasets[dataset]
    global X, y, s
    X = adult["X"].iloc[0:n_samples] #[["workclass", "marital-status", "occupation", "relationship", "native-country"]] #[["fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]
    y = adult["y"].iloc[0:n_samples]
    s = adult["s"].iloc[0:n_samples]
    #for col in s.keys():
    #    print(col)
    #    print(np.unique(s[col]))

    print("Dataset:", dataset)
    print("Amount of samples:", n_samples)
    print("Amount of estimators:", n_estimators,"\n")
    print("Starting... \n")

    ctree, ctreetime = IAFCTree()
    print("C++ fit:", ctreetime)
    ptree, ptreetime = IAFPythonTree()
    print("Python fit:", ptreetime)
    o1tree, o1time = IAFCalcoptTree()
    print("calc opt fit:", o1time)
    o2tree, o2time = IAFOHEoptTree()
    print("OHE opt fit:", o2time)




    cpred, cpredtime = PPCTree(ctree)
    print("C++ predict:", cpredtime)
    ppred, ppredtime = PPPythonTree(ptree)
    print("Python predict:", ppredtime)
    print("Trees preddiff:", PredDiff(ppred, cpred))
    
    print()

    pmult, pmulttime = IAFPythonForest(-1)
    print("Python Forest fit, n_jobs = -1:", pmulttime)
    psing, psingtime = IAFPythonForest(1)
    print("Python Forest fit, n_jobs = 1:", psingtime)
    
    cmult, cmulttime = IAFCForest(-1)
    print("C++ Forest fit, n_jobs = -1:", cmulttime)
    csing, csingtime = IAFCForest(1)
    print("C++ Forest fit, n_jobs = 1:", csingtime)

    hmult, hmulttime = IAFHybridForest(-1)
    print("Hybrid Forest fit, n_jobs = -1:", hmulttime)
    hsing, hsingtime = IAFHybridForest(1)
    print("Hybrid Forest fit, n_jobs = 1:", hsingtime)

    print()

    pmultpred, pmultpredtime = PPPythonForest(pmult)
    print("Python Forest predict, n_jobs = -1:", pmultpredtime)
    psingpred, psingpredtime = PPPythonForest(psing)
    print("Python Forest predict, n_jobs = 1", psingpredtime)
    cmultpred, cmultpredtime = PPCForest(cmult)
    print("C++ Forest predict, n_jobs = -1:", cmultpredtime)
    csingpred, csingpredtime = PPCForest(csing)
    print("C++ Forest predict, n_jobs = 1:", csingpredtime)
    hmultpred, hmultpredtime = PPCForest(hmult)
    print("Hybrid Forest predict, n_jobs = -1:", hmultpredtime)
    hsingpred, hsingpredtime = PPCForest(hsing)
    print("Hybrid Forest predict, n_jobs = 1:", hsingpredtime)

    print()

    p2 = FDTC.FairRandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=43)
    p2.fit(X=X, y=y, s=s)
    otherpred = p2.predict_proba(X)[:,1]

    print("Random States preddiff:", PredDiff(otherpred, pmultpred))
    print("Inter-Python preddiff:", PredDiff(pmultpred, psingpred))
    print("Inter-C++ preddiff:", PredDiff(cmultpred, csingpred))
    print("Inter-Hyrbid preddiff:", PredDiff(hmultpred, hsingpred))
    print("Python vs C++ preddiff:", PredDiff(pmultpred, cmultpred))
    print("Python vs Hybrid preddiff:", PredDiff(pmultpred, hmultpred))
    print("C++ vs Hybrid preddiff:", PredDiff(cmultpred, hmultpred))

    print()

    print("Python MSE:", MSE(pmultpred))
    print("C++ MSE:", MSE(cmultpred))
    print("Hybrid MSE:", MSE(hmultpred))
    CI = ((np.ones(len(y)) == y).sum() / len(y))
    print("Class imbalance:", CI)
    
    #X, y, s = tools.load("large_num_complex")

    #tree = FDTC.FairDecisionTreeClassifier(orthogonality=0)
    
    #original_scores = cross_val_score(tree, X, y, scoring="roc_auc", verbose=5, error_score="raise", fit_params={"s": s}, n_jobs=-1)

    #datasets = joblib.load('datasets.pkl')
    #adult = datasets[dataset]
    #X = adult["X"]
    #y = adult["y"]
    #s = adult["s"]
    #benchmark_scores = cross_val_score(tree, X, y, scoring="roc_auc", verbose=5, error_score="raise", fit_params={"s": s}, n_jobs=-1)
    #print("generated:", generated_scores)
    #print("benchmark:", benchmark_scores)

if __name__ == "__main__":
    Main()



