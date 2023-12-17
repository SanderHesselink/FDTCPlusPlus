# FDTC++

This is the GitHub repo for the FDTC++, a C++ implementation of the [Fair Decision Tree Classifier](https://github.com/pereirabarataap/fair_tree_classifier) by [@pereirabarataap](https://github.com/pereirabarataap). 

The repo consists of:
- The C++ implementation, contained in `module.cpp`
- A trimmed version of the original Python implementation, `FDTC.py`
- Two adjusted versions of the Python implementation
- Several experiments comparing the different implementations, mostly in terms of runtime, but also some output validation


## How to use
After installing the dependencies and choosing a compiler (see below), running `main.py` will run all experiments consecutively. These experiments use synthetic datasets, as well as some fairness benchmark datasets, contained in `datasets.pkl`.

To use the model on your own data, only the `module.cpp` file is required. This contains two classes: the Fair Decision Tree Classifier (FDTC) and the Fair Random Forest Classifier (FRFC), which can be called from Python.
 ### Example
 ```Python
import FDTCPP
import joblib

data = joblib.load(datasets.pkl)["adult"
X = data["X"]
y = data["y"]
s = data["s"]

clf = FDTCPP.FDTC()
clf.fit(X, y, s)
y_prob = clf.predict_proba(X)[:,1]
```


## Dependencies

This project relies on the pybind11 library in order to make C++ code callable from Python. A guide on how to install this library can be found in the [pybind11 docs](https://pybind11.readthedocs.io/en/stable/installing.html). Alternatively, if you're using Visual Studio, Microsoft has published a [more specialised tutorial](https://learn.microsoft.com/en-us/visualstudio/python/working-with-c-cpp-python-in-visual-studio). Keep in mind that the module name should be "FDTCPP" (all caps), so adjust the names used in the tutorials accordingly. 

Besides pybind11, all C++ dependencies are part of the standard library.

The Python code requires the following modules:
- numpy
- pandas
- multiprocessing
- scipy
- joblib
- sklearn


## Compiler

We highly recommend compiling the C++ code using an Intel compiler, available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) as a standalone version, or [here](https://marketplace.visualstudio.com/items?itemName=intel-corporation.dpcpponline) as a Visual Studio extension.
Additionally, due to differences in compilers across platforms, we are aware that the C++ code may not compile with GCC.
