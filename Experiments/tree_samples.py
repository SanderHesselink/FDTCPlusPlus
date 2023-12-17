import tools
import FDTC
import Calc_opt
import OHE_opt
import FDTCPP

import numpy as np
import pandas as pd
import time
import pickle
    
        
def vary_samples_c(name):
    resdict = {}

    X, y, s = tools.load(name)

    tree = FDTCPP.FDTC(min_samples_leaf=1, min_samples_split=2)

    resdict["samples"] = list(range(1, int(len(X[0])/10), int(len(X[0])/100)))
    resdict["samples"].append(10000)
    print(resdict["samples"])
    for i in range(len(X)):
        print(i)
        result = []
        for n in resdict["samples"]:
            print(n)
            dataX = X[i][:n]
            datay = y[i][:n]
            datas = s[i][:n]

            t0 = time.perf_counter()
            tree.fit(dataX, datay, datas)
            t1 = time.perf_counter()
            result.append(t1-t0)
        resdict[i] = result

    savename = "Data/Results/vary_samples_c_"+name+ str(time.time())[:10]+".pkl"

    with open(savename, 'wb') as fp:
        pickle.dump(resdict, fp)
        
    return savename

def vary_samples_python(name, opt="regular"):
    resdict = {}

    X, y, s = tools.load(name)

    if (opt == "OHE"):
        tree = OHE_opt.FairDecisionTreeClassifier()
    elif (opt == "calc"):
        tree = Calc_opt.FairDecisionTreeClassifier()
    elif (opt == "regular"):
        tree = FDTC.FairDecisionTreeClassifier()
    else:
        raise ValueError("Invalid argument")
    
    resdict["samples"] = list(range(1, len(X[0]), int(len(X[0])/10)))
    for i in range(len(X)):
        print(i)
        result = []
        for n in range(1, len(X[i]), int(len(X[i])/10)):
            print(n)
            dataX = X[i][:n]
            datay = y[i][:n]
            datas = s[i][:n]

            t0 = time.perf_counter()
            tree.fit(dataX, datay, datas)
            t1 = time.perf_counter()
            result.append(t1-t0)
        resdict[i] = result
            

    savename = "Data/Results/vary_samples_python_"+str(opt)+"_"+name+ str(time.time())[:10]+".pkl"

    with open(savename, 'wb') as fp:
        pickle.dump(resdict, fp)
   
    return savename

