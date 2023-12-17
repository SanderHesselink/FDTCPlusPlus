# FDTC++

This is the GitHub repo for the FDTC++, a C++ implementation of the [Fair Decision Tree Classifier](https://github.com/pereirabarataap/fair_tree_classifier/blob/main/fair_trees.py) by [@pereirabarataap](https://github.com/pereirabarataap). 

The repo consists of:
- The C++ implementation, contained in module.cpp
- A trimmed version of the original Python implementation, FDTC.py
- Two adjusted versions of the Python implementation
- Several experiments comparing the different implementations

If you wish to use the algorithm, and are not interested in any experiments or comparisons, only the file module.cpp is required.


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
