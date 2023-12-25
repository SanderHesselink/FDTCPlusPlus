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

data = joblib.load(datasets.pkl)["adult"]
# The model only works on Python lists and NumPy arrays, so pandas DataFrames need to be converted first
X = data["X"].to_numpy()
y = data["y"].to_numpy()
s = data["s"].to_numpy()

clf = FDTCPP.FDTC()
clf.fit(X, y, s)
y_prob = clf.predict_proba(X)[:,1]
```
### Note about using the Fair Random Forest Classifier
Unfortunately, the FRFC shows poor execution speed when paralellism is enabled. As such, it is recommended to use the Hybrid Forest, contained in the file `ctpf.py`. This wraps the C++ tree in a Python forest, and is significantly faster than the pure C++ forest when paralellism is enabled.


## Dependencies

This project relies on the pybind11 library in order to make C++ code callable from Python. A guide on how to install this library can be found in the [pybind11 docs](https://pybind11.readthedocs.io/en/stable/installing.html). Alternatively, if you're using Visual Studio, Microsoft has published a [more specialised tutorial](https://learn.microsoft.com/en-us/visualstudio/python/working-with-c-cpp-python-in-visual-studio). Keep in mind that the Python code in this repo expects the module name to be "FDTCPP" (all caps), so adjust the names used in the tutorials accordingly. 

Besides pybind11, all C++ dependencies are part of the standard library.

The Python code requires the following modules:
- numpy
- pandas
- multiprocessing
- scipy
- joblib
- sklearn


## Compiler

We highly recommend compiling the C++ code using an Intel compiler, available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) as a standalone version, or [here](https://marketplace.visualstudio.com/items?itemName=intel-corporation.dpcpponline) as a Visual Studio extension. This compiler has heavy optimizations for the C++ `valarray` class, which is used extensively in this module.
Additionally, due to differences in compilers across platforms, we are aware that the C++ code may not compile with GCC.
