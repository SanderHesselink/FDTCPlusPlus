import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from sklearn.model_selection import KFold

import datasets
import superfastcode
import FDTC
import math


n_range = list(range(1000, 10001, 1000))
m_range = list(range(10, 101, 10))

def heatmap_data(s_shift, lang):
    result = []
    
    for n in n_range:
        result.insert(0, [])
        for m in m_range:
            print(n, m)
            X, y, s = datasets.complex_variable(n, m, s_shift)
            set_times = np.zeros(5)
            for i in range(5):
                kf = KFold()
                fold_times = np.zeros(5)
                for index, (train, test) in enumerate(kf.split(X[i])):
                    trainX = X[i][train]
                    trainy = y[i][train]
                    trains = s[i][train]
                    testX = X[i][test]
                    if lang == "Python":
                        tree = FDTC.FairDecisionTreeClassifier()
                    else:
                        tree = superfastcode.FDTC()
                    t0 = perf_counter()
                    tree.fit(trainX, trainy, trains)
                    tree.predict_proba(testX)
                    t1 = perf_counter()
                    fold_times[index] = t1 - t0
                    break
                set_times[i] = fold_times[0]
            result[0].append(set_times.mean())
    return(result)

def draw_heatmap(data, cbar=True, axis=None, crange=None, annotate="ratio", cmap=None, threshold=0.6):
    matrix = np.array(data)
    #print(math.floor((matrix.sum() * 5) / 3600), "hours and", round(((matrix.sum() * 5) % 3600) / 60), "minutes")
    
    if not axis:
        fig, ax = plt.subplots()
    else:
        ax = axis
    if not crange:
        vmin = matrix.min()
        vmax = matrix.max()
    else:
        vmin = min(crange)
        vmax = max(crange)
    if not cmap:
        cmap = "Blues"
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(10), labels=list(range(1, 11, 1)))
    ax.set_yticks(range(10)[::-1], labels=m_range)
    ax.set_ylabel("Number of features")
    ax.set_xlabel("Number of samples (x1000)")
    if annotate:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if round((matrix[i][j] - vmin) / (vmax - vmin), 2) < threshold:
                    color = "black"
                else:
                    color = "white"
                if annotate == "ratio":
                    value = round(matrix[i][j] / vmax, 2)
                else:
                    value = round(matrix[i][j], annotate)
                text = ax.text(j, i, value, color=color, ha="center", va="center", font={"size":6})
    if cbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("runtime (seconds)", rotation=-90, va="bottom")
        maxlabel = ax.annotate(f"{round(matrix.max(), 2)}", xy=(9, 0), xytext=(11, -0.5))
        minlabel = ax.annotate("0", xy=(9, 9), xytext=(11, 9.7))
    if not axis:
        fig.tight_layout()
    return im

def heatmap_comparison(data1, data2, shared=True, compare="ratio"):
    matrix1 = np.array(data1)
    matrix2 = np.array(data2)
    maxval = round(max(matrix1.max(), matrix2.max()), 2)
    minval = round(min(matrix1.min(), matrix2.min()), 2)
    if shared:
        crange = [minval, maxval]
    else:
        crange = None
    figure = plt.figure(figsize=[10,10])
    #gs = figure.add_gridspec(2, 3, height_ratios=[20,1], hspace=-0.3)
    gs = figure.add_gridspec(4, 4, height_ratios=[20,1,1,20], hspace=0.5, width_ratios=[20,1,20,1])
    axis = [[figure.add_subplot(gs[0, 0]),
        figure.add_subplot(gs[0, 2])],
        [figure.add_subplot(gs[1, 0:3])],
        [figure.add_subplot(gs[3, 0]),
        figure.add_subplot(gs[3, 2])]]

    #if compare == "ratio":
    #    im1 = draw_heatmap(matrix1, axis=axis[0][0], cbar=False, crange=crange)
    #    im2 = draw_heatmap(matrix2, axis=axis[0][1], cbar=False, crange=crange)
    #    draw_heatmap(matrix2 / matrix1, axis=axis[0][2], cbar=False, cmap="Greens", threshold=0.65, annotate=2)
    #elif compare == "difference":
    im1 = draw_heatmap(matrix1, axis=axis[0][0], cbar=False, crange=crange, annotate=1)
    im2 = draw_heatmap(matrix2, axis=axis[0][1], cbar=False, crange=crange, annotate=1)
    draw_heatmap(abs(matrix2 - matrix1), axis=axis[2][0], cbar=False, cmap="Oranges", threshold=0.65, annotate=1)
    draw_heatmap((matrix2 / matrix1), axis=axis[2][1], cbar=False, cmap="Greens", annotate=1)
    #axis[2][1].plot(m_range, (matrix2 / matrix1)[::-1])
    #axis[2][1].legend(n_range, title="Number of\n samples", loc=[1,0])

    #axis[0][2].set_title(compare)

    #axis[0][1].set_ylabel("")
    #axis[0][2].set_ylabel("")
    #axis[0][0].set_xlabel("")
    #axis[0][2].set_xlabel("")
    #axis[2][1].set_xlabel("Number of features")
    #axis[2][1].set_ylabel("Relative speedup")
    cbar = axis[1][0].figure.colorbar(im2, cax=axis[1][0], orientation="horizontal")
    cbar.ax.set_xlabel("Runtime in seconds")
    #maxlabel = axis[1][0].annotate(f"{maxval}", xy=(1, 0), xytext=(maxval, -0.6), ha="center")
    #minlabel = axis[1][0].annotate("0", xy=(1, 0), xytext=(0, -0.6), ha="center")
    
    return axis