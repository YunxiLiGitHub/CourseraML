MachineLearning的完整实现，与网易课程完全同步。

有四个目录:
- Jupyter:基于jupyter的版本，使用的是python2
- Python:基于Python的版本。
- Advanced(NOTREMOVE):基于Python实现，非常好的总结性实现，等到全部学完之后，使用这个复习，绝对不能删除。
- MyCode:基于jupyter版本和Python版本，自己生成的，基于Python的实现。基本思路是，使用基于Python的版本的总体框架，参考jupyter的版本的具体实现。

自己的代码在MyCode目录下:
- 使用的IDE是LiClipse。
- 每个Ex建立各自的project。

## How to start
### Dependencies
This project was coded in Python 3.6
* numpy
* matplotlib
* scipy
* scikit-learn
* scikit-image
* nltk

### Installation
The fastest and easiest way to install all these dependencies at once is to use [Anaconda](https://www.continuum.io/downloads).


## Important Note
There are a couple of things to keep in mind before starting.
* all column vectors from octave/matlab are flattened into a simple 1-dimensional ndarray. (e.g., y's and thetas are no longer m x 1 matrix, just a 1-d ndarray with m elements.)
So in Octave/Matlab, 
    ```matlab
    >> size(theta)
    >> (2, 1)
    ```
    Now, it is
    ```python
    >>> theta.shape
    >>> (2, )
    ```
* numpy.matrix is never used, just plain ol' numpy.ndarray

## Contents
#### [Exercise 1](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex1)
* Linear Regression
* Linear Regression with multiple variables
#### [Exercise 2](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex2)
* Logistic Regression
* Logistic Regression with Regularization
#### [Exercise 3](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex3)
* Multiclass Classification
* Neural Networks Prediction fuction
#### [Exercise 4](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex4)
* Neural Networks Learning
#### [Exercise 5](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex5)
* Regularized Linear Regression
* Bias vs. Variance
#### [Exercise 6](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex6)
* Support Vector Machines
* Spam email Classifier
#### [Exercise 7](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex7)
* K-means Clustering
* Principal Component Analysis
#### [Exercise 8](https://github.com/nsoojin/coursera-ml-py/tree/master/machine-learning-ex8)
* Anomaly Detection
* Recommender Systems

## Solutions
You can check out my implementation of the assignments [here](https://github.com/nsoojin/coursera-ml-py-sj). I tried to vectorize all the solutions.