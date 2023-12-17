import tools
import FDTC
import ctpf
import FDTCPP
import Calc_opt
import OHE_opt
import tree_samples
import datasets
import heatmap
import pickle
import numpy as np
import os
import time
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import multiprocessing
import joblib


    
cdict = {"cpp" : "tab:blue", "python" : "tab:orange", "hybrid" : "tab:green"}

def QA_preddiff_trees():
    langs = ["cpp", "calc", "ohe"]

    preddict = {}
    probpreddict = {}
    X, y, s = datasets.complex_variable(1000, 10, 0.1)
    for i in range(5):
        print(i)
        kf = KFold()
        for index, (train, test) in enumerate(kf.split(X[i])):
            print(index)
            trainX = X[i][train]
            trainy = y[i][train]
            trains = s[i][train]
            testX = X[i][test]
            tree = FDTC.FairDecisionTreeClassifier()
            tree.fit(X=trainX, y=trainy, s=trains)
            probpreddict[(i, index)] = np.array(tree.predict_proba(X=testX))
            preddict[(i, index)] = np.array(tree.predict(X=testX))


    for lang in langs:
        result = []
        probresult = []
        print(lang)
        for i in range(5):
            for index, (train, test) in enumerate(kf.split(X[i])):
                trainX = X[i][train]
                trainy = y[i][train]
                trains = s[i][train]
                testX = X[i][test]
                if lang == "calc":
                    tree = Calc_opt.FairDecisionTreeClassifier()
                elif lang == "ohe":
                    tree = OHE_opt.FairDecisionTreeClassifier()
                elif lang == "cpp":
                    tree = FDTCPP.FDTC()
                tree.fit(X=trainX, y=trainy, s=trains)
                probpred = np.array(tree.predict_proba(X=testX))
                pred = np.array(tree.predict(X=testX))
                result.append((preddict[(i, index)] != pred).sum())
                probresult.append(abs(probpreddict[(i, index)] - probpred).mean())
        print(np.array(result).sum())
        print(np.array(probresult).mean())

def QA_preddiff_forests():
    langs = ["python", "cpp", "hybrid"]

    preddict = {}
    probpreddict = {}
    X, y, s = datasets.complex_variable(1000, 10, 0.1)
    for i in range(5):
        print(i)
        kf = KFold()
        for index, (train, test) in enumerate(kf.split(X[i])):
            print(index)
            trainX = X[i][train]
            trainy = y[i][train]
            trains = s[i][train]
            testX = X[i][test]
            tree = FDTC.FairRandomForestClassifier(n_jobs=-1)
            tree.fit(X=trainX, y=trainy, s=trains)
            probpreddict[(i, index)] = np.array(tree.predict_proba(X=testX))
            preddict[(i, index)] = np.array(tree.predict(X=testX))


    for lang in langs:
        result = []
        probresult = []
        print(lang)
        for i in range(5):
            for index, (train, test) in enumerate(kf.split(X[i])):
                trainX = X[i][train]
                trainy = y[i][train]
                trains = s[i][train]
                testX = X[i][test]
                if lang == "python":
                    tree = FDTC.FairRandomForestClassifier(random_state=43, n_jobs=-1)
                elif lang == "hybrid":
                    tree = ctpf.FairRandomForestClassifier(n_jobs=-1)
                elif lang == "cpp":
                    tree = FDTCPP.FRFC()
                tree.fit(X=trainX, y=trainy, s=trains)
                probpred = np.array(tree.predict_proba(X=testX))
                pred = np.array(tree.predict(X=testX))
                result.append((preddict[(i, index)] != pred).sum())
                probresult.append(abs(probpreddict[(i, index)] - probpred).mean())
        print(np.array(result).sum())
        print(np.array(probresult).mean())

def four_trees_table():
    langs = ["python", "cpp", "calc", "ohe"]
    for lang in langs:
        name = f"Data/Results/table_n1_m1_s01_{str(time.time())[:10]}_{lang}.pkl"
        result = []
        X, y, s = datasets.complex_variable(5000, 50, 0.1)
        for i in range(5):
            kf = KFold()
            for index, (train, test) in enumerate(kf.split(X[i])):
                trainX = X[i][train]
                trainy = y[i][train]
                trains = s[i][train]
                testX = X[i][test]
                if lang == "python":
                    tree = FDTC.FairDecisionTreeClassifier()
                elif lang == "calc":
                    tree = Calc_opt.FairDecisionTreeClassifier()
                elif lang == "ohe":
                    tree = OHE_opt.FairDecisionTreeClassifier()
                elif lang == "cpp":
                    tree = FDTCPP.FDTC()
                t0 = perf_counter()
                tree.fit(X=trainX, y=trainy, s=trains)
                t1 = perf_counter()
                tree.predict_proba(X=testX)
                t2 = perf_counter()
                fold_times = [t1-t0, t2-t1]
                result.append(fold_times)
        result = np.array(result)
        print(lang)
        print((result[:,0] + result[:,1]).mean(), (result[:,0] + result[:,1]).std())
        print(result[:,0].mean(), result[:,1].mean())

        with open(name, "wb") as fp:
            pickle.dump(result, fp)

    for file in os.listdir("Data/Results"):
        if "table" in file:
            print(file)
            with open("Data/Results/" + file, "rb") as fp:
                data = pickle.load(fp)
            data = np.array(data)
            #print("fit:", data[:,0].mean(), ",", data[:,0].std())
            #print("predict:", data[:,1].mean(), ",", data[:,1].std())
            print("total:", (data[:,0] + data[:,1]).mean(), ",", (data[:,0] + data[:,1]).std())



def S_shift_table():
    result = []
    for lang in ["Python", "C++"]:
        result.append([])
        for s_shift in [0.1, 0.2, 0.3, 0.4, 0.5]:
            X, y, s = datasets.complex_variable(10000, 100, s_shift)
            set_times = []
            for i in range(5):
                kf = KFold()
                fold_times = []
                for index, (train, test) in enumerate(kf.split(X[i])):
                    print(s_shift, i, index)
                    trainX = X[i][train]
                    trainy = y[i][train]
                    trains = s[i][train]
                    testX = X[i][test]
                    if lang == "Python":
                        tree = FDTC.FairDecisionTreeClassifier()
                    else:
                        tree = FDTCPP.FDTC()
                    t0 = perf_counter()
                    tree.fit(trainX, trainy, trains)
                    tree.predict_proba(testX)
                    t1 = perf_counter()
                    fold_times.append(t1 - t0)
                set_times.append(fold_times)
            result[-1].append(set_times)

    name = "Data/Results/heatmap_s_shift_" + str(time.time())[:10] +"_python.pkl"

    with open(name, "wb") as fp:
        pickle.dump(result, fp)


def AUC_table():
    n_range = [10000]
    m_range = [100]

    result = []
    
    for n in n_range:
        for m in m_range:
            print(n, m)
            X, y, s = datasets.complex_variable(n, m, 0.5)
            set_times = np.zeros(5)
            for i in range(5):
                kf = KFold()
                fold_times = np.zeros(5)
                for index, (train, test) in enumerate(kf.split(X[i])):
                    trainX = X[i][train]
                    trainy = y[i][train]
                    trains = s[i][train]
                    testX = X[i][test]
                    testy = y[i][test]
                    #tree = FDTCPP.FDTC(max_depth=-1)
                    tree = FDTC.FairDecisionTreeClassifier()
                    tree.fit(trainX, trainy, trains)
                    tree.predict_proba(testX)
                    pred = tree.predict(testX)
                    score = roc_auc_score(testy, pred)
                    result.append(score)

    result = np.array(result)
    print(result.mean(), result.std())

def Heatmap_data():
    s_shifts = np.array(list(range(2, 6, 1))) / 10

    for s_shift in s_shifts:
        print("C++", s_shift)
        data = heatmap.heatmap_data(s_shift, "C++")
        name = "Data/Results/heatmap_nbins" + str(s_shift)[-1] + "_" + str(time.time())[:10] +"_cpp.pkl"
        print(name)
        with open(name, 'wb') as fp:
            pickle.dump(data, fp)


def n_bins_data():
    lang = "C++"
    result = []
    for n in np.array(range(500, 5001, 500)):
        for m in [50]:
            print(n, m)
            X, y, s = datasets.complex_variable(5000, m, 0.1)
            set_times = []
            for i in range(5):
                if lang == "Python":
                    tree = FDTC.FairDecisionTreeClassifier(n_bins=n)
                else:
                    tree = FDTCPP.FDTC(n_bins=n)
                t0 = perf_counter()
                tree.fit(X[i], y[i], s[i])
                t1 = perf_counter()
                set_times.append(t0 - t1)
            result.append(set_times)
    name = "Data/Results/heatmap_nbins5_" + str(time.time())[:10] +"_cpp.pkl"
    with open(name, "wb") as fp:
        pickle.dump(result, fp)


def Plot_n_bins():
    figure = plt.figure(figsize=[16,6])
    gs = figure.add_gridspec(1, 2, hspace=0.3)
    axis = [figure.add_subplot(gs[0,0]), figure.add_subplot(gs[0, 1])]
    n_range = range(500, 5001, 500)
    name = "Data/Results/heatmap_nbins5_1696168036_python.pkl"
    with open(name, "rb") as fp:
        data1 = pickle.load(fp)
    data1 = np.array([conf.mean() for conf in np.array(data1) * -1]) #t0 - t1 should've been t1 - t0, woops
    axis[0].plot(n_range, data1, color="tab:orange")

    name = "Data/Results/heatmap_nbins5_1696174771_cpp.pkl"
    with open(name, "rb") as fp:
        data2 = pickle.load(fp)  
    data2 = np.array([conf.mean() for conf in np.array(data2) * -1])
    axis[0].plot(n_range, data2, color="tab:blue")
    axis[0].set_xticks(n_range)
    axis[0].set_xlabel("Number of unique values", font={"size":15})
    axis[0].set_ylabel("Runtime in seconds", font={"size":15})
    axis[0].legend(["Python", "C++"])
    axis[0].set_title("Runtimes", font={"size":15})
    axis[1].plot(n_range, data1 / data2, color="tab:green")
    axis[1].set_xticks(n_range)
    axis[1].set_yticks(range(5))
    axis[1].set_xlabel("Number of unique values", font={"size":15})
    axis[1].set_ylabel("Relative speedup", font={"size":15})
    axis[1].set_title("Ratio", font={"size":15})
    plt.tight_layout()
    plt.show()


      
def Plot_heatmaps
    n_range = list(range(1000, 10001, 1000))
    m_range = list(range(10, 101, 10))
    with open(CPP[1], 'rb') as fp:
        data1 = np.array(pickle.load(fp))

    with open(Python[1], "rb") as fp:
        data2 = np.array(pickle.load(fp))

    ax = heatmap.heatmap_comparison(data1, data2, compare="difference")
    ax[0][0].set_title("C++")
    ax[0][1].set_title("Python")
    ax[2][0].set_title("Difference")
    ax[2][1].set_title("Ratio")
    plt.suptitle("S shift = 0.1")
    plt.show()



def Plot_s_shift():
    xlabels = np.array(range(1, 6)) / 10
    name = "Data/Results/compare_s_shift_1696084683_python.pkl"

    with open(name, "rb") as fp:
        data = pickle.load(fp)


    matrix1 = np.array([np.mean(n) for n in data[0]])
    matrix2 = np.array([np.mean(n) for n in data[1]])

    figure = plt.figure(figsize=(16, 6))
    gs = figure.add_gridspec(1, 2, hspace=0.3)
    axis = [
        figure.add_subplot(gs[0, 0]),
        figure.add_subplot(gs[0, 1])
        ]

    colors = ["tab:orange", "tab:blue"]


    
    axis[0].plot(xlabels, matrix1, color="tab:orange")
    axis[0].plot(xlabels, matrix2, color="tab:blue")
    axis[0].legend(["Python", "C++"])
    for n in range(2):
        for i in range(5):
            for j in range(5):
                for val in data[n][i][j]:
                    axis[0].scatter((i+1)/10, val, color=colors[n], alpha=0.3)
    axis[0].set_xticks(xlabels)
    axis[0].set_xlabel("S shift", font={"size": 15})
    axis[0].xaxis.set_label_coords(1.1, -0.07)
    axis[0].set_ylabel("Runtime in seconds", font={"size": 15})
    axis[0].set_title("Runtimes", font={"size": 15})

    axis[1].plot(xlabels, matrix1/matrix2, color="tab:green")
    axis[1].set_xticks(xlabels)
    axis[1].set_yticks([0.5, 1, 1.5, 2])
    axis[1].set_ylabel("Relative speedup", font={"size": 15})
    axis[1].set_title("Ratio", font={"size": 15})
    plt.show()

def Forest_data():
    langs = ["python", "cpp", "hybrid"]
    n_range = [10000]
    jobs_range = [1]
    est_range = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    m = 10
    s_shift = 0.1

    for lang in langs:
        for n in n_range:
            for n_jobs in jobs_range:
                if n_jobs != 1 or lang != "python":
                    continue
                for n_estimators in est_range:
                    name = f"Data/Results/forest_{n}_{n_jobs}_{n_estimators}_{str(time.time())[:10]}_{lang}.pkl"
                    result = []
                    X, y, s = datasets.complex_variable(n, m, s_shift)
                    for i in range(5):
                        kf = KFold()
                        for index, (train, test) in enumerate(kf.split(X[i])):
                            print(name, i, index)
                            trainX = X[i][train]
                            trainy = y[i][train]
                            trains = s[i][train]
                            testX = X[i][test]
                            if lang == "python":
                                forest = FDTC.FairRandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators)
                            elif lang == "cpp":
                                forest = FDTCPP.FRFC(n_jobs=n_jobs, n_estimators=n_estimators)
                            elif lang == "hybrid":
                                forest = ctpf.FairRandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators)
                            t0 = perf_counter()
                            forest.fit(X=trainX, y=trainy, s=trains)
                            t1 = perf_counter()
                            forest.predict_proba(X=testX)
                            t2 = perf_counter()
                            fold_times = [t1-t0, t2-t1]
                            break
                        result.append(fold_times)

                    with open(name, "wb") as fp:
                        pickle.dump(result, fp)

def Plot_forest_comparison():
    est_range = list(range(50, 501, 50))
    est_range.insert(0, 1)
    langs = ["python", "cpp", "hybrid"]
    n = 10000
    n_jobs = [1, -1]
    s = 1

    figure = plt.figure(figsize=(16,4))
    gs = figure.add_gridspec(1, 3, hspace=0.3)
    axis = [
        [figure.add_subplot(gs[0, 0]),
        figure.add_subplot(gs[0, 1]),
        figure.add_subplot(gs[0, 2])],
        ]

    for lang in langs:
        job_dict = {}
        for n_job in n_jobs:
            result = []
            for est in est_range:
                part = f"forest_{n}_{n_job}_{est}_{s}"
                for file in os.listdir("Data/Results"):
                    if part in file and lang in file:
                        with open(f"Data/Results/{file}", "rb") as fp:
                            data = pickle.load(fp)
                            result.append(data)
            fit = [run[0] for res in result for run in res]
            predict = [run[1] for res in result for run in res]

            fit = np.array([i.mean() for i in np.array_split(fit, 11)])
            predict = np.array([i.mean() for i in np.array_split(predict, 11)])
            job_dict[n_job] = predict
        axis[0][0].plot(est_range, job_dict[1], color=cdict[lang])
        axis[0][1].plot(est_range, job_dict[-1], color=cdict[lang])
        axis[0][2].plot(est_range, job_dict[1] / job_dict[-1], color=cdict[lang])
            
    
    axis[0][0].set_title("No parallel")
    axis[0][1].set_title("Parallel")
    axis[0][2].set_title("Ratio")
    axis[0][1].set_xlabel("Number of estimators")
    axis[0][0].set_ylabel("Runtime in seconds")
    axis[0][1].set_ylabel("Runtime in seconds")
    axis[0][2].set_ylabel("Relative speedup")
    axis[0][0].legend(["Python", "C++", "Hybrid"])


    plt.tight_layout()
    plt.show()


def Forest_prediction_table:
    est_range = list(range(50, 501, 50))
    langs = ["python", "cpp", "hybrid"]
    n_jobs = [1, -1]

    for lang in langs:
        for n_job in n_jobs:
            for n in [1000, 10000]:
                result = []
                for est in est_range:
                    part = f"forest_{n}_{n_job}_{est}_1"
                    for file in os.listdir("Data/Results"):
                        if part in file and lang in file:
                            with open(f"Data/Results/{file}", "rb") as fp:
                                data = pickle.load(fp)
                                result.append(data)
                predict = [run[1] for res in result for run in res]

                predict = np.array([i.mean() for i in np.array_split(predict, 11)])
                print(lang, n, n_job)
                print(predict[-1])

def RWD_table_trees():
    keys = ["adult", "dutch_census", "german_credit", "bank_marketing", "law_school"]
    datadict = joblib.load('datasets.pkl')
    
    for lang in ["python", "cpp", "calc", "ohe"]:
        for key in keys:
            dataset = datadict[key]
            X = dataset["X"]
            y = dataset["y"]
            s = dataset["s"]
            result = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):
                trainX = X.iloc[train]
                trainy = y.iloc[train]
                trains = s.iloc[train]
                testX = X.iloc[test]
                testy = y.iloc[test]
                if lang == "python":
                    tree = FDTC.FairDecisionTreeClassifier()
                    t0 = perf_counter()
                    tree.fit(X=trainX, y=trainy, s=trains)
                    t1 = perf_counter()
                    tree.predict_proba(testX)
                    t2 = perf_counter()
                elif lang == "calc":
                    tree = Calc_opt.FairDecisionTreeClassifier()
                    t0 = perf_counter()
                    tree.fit(X=trainX, y=trainy, s=trains)
                    t1 = perf_counter()
                    tree.predict_proba(testX)
                    t2 = perf_counter()
                elif lang == "ohe":
                    tree = OHE_opt.FairDecisionTreeClassifier()
                    t0 = perf_counter()
                    tree.fit(X=trainX, y=trainy, s=trains)
                    t1 = perf_counter()
                    tree.predict_proba(testX)
                    t2 = perf_counter()
                elif lang == "cpp":
                    tree = FDTCPP.FDTC()
                    t0 = perf_counter()
                    tree.fit(X=trainX.to_numpy(), y=trainy.to_numpy(), s=trains.to_numpy())
                    t1 = perf_counter()
                    tree.predict_proba(testX.to_numpy())
                    t2 = perf_counter()
                result.append([t1 - t0, t2-t1])
            print(lang, key)
            print(np.array(result)[:,0].mean(), np.array(result)[:,1].mean())

            name = f"Data/Results/rwd_{key}_{lang}.pkl"
            with open(name, 'wb') as fp:
                pickle.dump(result, fp)

def RWD_table_forests():
    keys = ["adult", "dutch_census", "german_credit", "bank_marketing", "law_school"]
    datadict = joblib.load('datasets.pkl')
    
    for lang in ["hybrid"]:
        for key in keys:
            dataset = datadict[key]
            X = dataset["X"]
            y = dataset["y"]
            s = dataset["s"]
            result = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):
                trainX = X.iloc[train]
                trainy = y.iloc[train]
                trains = s.iloc[train]
                testX = X.iloc[test]
                testy = y.iloc[test]
                if lang == "python":
                    tree = FDTC.FairRandomForestClassifier(n_jobs=-1, n_estimators=100)
                    t0 = perf_counter()
                    tree.fit(X=trainX, y=trainy, s=trains)
                    t1 = perf_counter()
                    tree.predict_proba(testX)
                    t2 = perf_counter()
                elif lang == "hybrid":
                    tree = ctpf.FairRandomForestClassifier(n_jobs=-1, n_estimators=100)
                    t0 = perf_counter()
                    tree.fit(X=trainX.to_numpy(), y=trainy.to_numpy(), s=trains.to_numpy())
                    t1 = perf_counter()
                    tree.predict_proba(testX.to_numpy())
                    t2 = perf_counter()
                elif lang == "cpp":
                    tree = FDTCPP.FRFC(n_jobs=-1, n_estimators=100)
                    t0 = perf_counter()
                    tree.fit(X=trainX.to_numpy(), y=trainy.to_numpy(), s=trains.to_numpy())
                    t1 = perf_counter()
                    tree.predict_proba(testX.to_numpy())
                    t2 = perf_counter()
                result.append([t1 - t0, t2-t1])
            print(lang, key)
            print(np.array(result)[:,0].mean(), np.array(result)[:,1].mean())

            name = f"Data/Results/rwdforest_{key}_{lang}.pkl"
            with open(name, 'wb') as fp:
                pickle.dump(result, fp)

    for name in os.listdir("Data/Results"):
        if "rwdforest" in name:
            print(name)
            with open(f"Data/Results/{name}", "rb") as fp:
                result = pickle.load(fp)
            print(np.array(result)[:,0].mean(), np.array(result)[:,1].mean())

def Plot_overfit():
    name = "Data/Results/vary_samples_c_big_one_d1694455266.pkl"
    with open(name, "rb") as fp:
        data1 = pickle.load(fp)

    name = "Data/Results/vary_samples_c_big_one_d1694455037.pkl"
    with open(name, "rb") as fp:
        data2 = pickle.load(fp)


    figure = plt.figure(figsize=(16, 6))
    gs = figure.add_gridspec(1, 2, hspace=0.3)
    axis = [
        figure.add_subplot(gs[0, 0]),
        figure.add_subplot(gs[0, 1])
        ]
    axis[0].plot(data1["samples"], data1[0])
    axis[1].plot(data2["samples"], data2[0])
    axis[0].set_ylabel("Runtime in seconds", font={"size": 15})
    axis[0].set_xlabel("Number of samples", font={"size": 15})
    axis[0].xaxis.set_label_coords(1.1, -0.07)
    axis[0].set_title("Default hyperparameters", font={"size": 15})
    axis[1].set_ylabel("Runtime in seconds", font={"size": 15})
    axis[1].set_title("Adjusted hyperparameters", font={"size": 15})
    plt.tight_layout()
    plt.show()
