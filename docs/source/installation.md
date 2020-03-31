# Installation Guide

## Installing Classification Report

Classification Report  library runs on python 3.6 and greater.

```shell
pip install classification-report
```

### Things that are good to know

classification-report library is written in pure python and depends on a few key python packages

1. [Pytorch](https://pytorch.org/), An open source machine learning framework that accelerates the path from research prototyping to production deployment.
2. [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html),TensorBoard provides the visualization and tooling needed for machine learning experimentation.
3. [Numpy](https://numpy.org/), NumPy is the fundamental package for scientific computing with Python.
4. [Seaborn](https://seaborn.pydata.org/), Seaborn is a Python data visualization library based on matplotlib.
5. [Matplotlib](https://matplotlib.org), Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
6. [scikit-learn](https://scikit-learn.org/stable/), Machine Learning in Python



## Installing from Source

```shell
git clone https://github.com/aman5319/Classification-Report
cd Classification-Report
```

```
pip install -e .
```



## Developing Classification Report

For developing classification report you can install the library with all development dependencies but first clone the repo.

```shell
git clone https://github.com/aman5319/Classification-Report
cd Classification-Report
```

```
pip install -e ".[dev]"
```



