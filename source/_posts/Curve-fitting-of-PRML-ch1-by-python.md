---
title: Note01 Curve Fitting of PRML Ch1
date: 2019-03-20 11:04:49
tags: PRML, python
---

## Overview

We will first introduce the traditional curve fitting problem, then we will implement these algorithms in python.

### TODO:(cankaoziliao)
- Pattern Recognition And Machine Learning
- [ctgk/PRML](https://github.com/ctgk/PRML)

------

## Curve Fitting Problem

### Introduce

Suppose we have a set of data (we called it training set): 
- Input observation: $\mathbf{x} = (x_1, x_2 \cdots x_N)^T$
- Output observation: $\mathbf{t} = (t_1, t_2 \cdots t_N)^T$

Our aim is to find the inner pattern to make precisely pridiction of $t$ given a new input $x$. 
We represent the pattern by 
$$y(x, \mathbf{w}) = w_0 + w_1 x + w_2 x^2 +\cdots + w_Mx^M  = \sum_{j = 0}^M w_j x^j$$
where $M$ is the order of the polynomial and we use $\mathbf{w} = (w_0, w_1, \cdots w_M)$ to represent the set of $w_i$. 
<a name="overfitting">Sooner</a> we will find the bigger of $M$, the more accurate the curve of polynomial $y(x,\mathbf{w})$ fits the given observation data.

But how can I know if the polynomial fits the pattern well?
We use $E(\mathbf{w}) = \frac{1}{2} \sum_{n = 1}^N[y(x_n,\mathbf{w}) - t_n]^2$ to measure the misfit between $y(x,\mathbf{w})$ and real world data(training data).
The $\frac{1}{2}$ is for computing convenience.

We called the $E(\mathbf{w})$ _error function_. And our goal is to minimize it since the small value TODO:(xianshile) the polynomial and the data have small difference, thus the polynomial fits the real world well.

Then let's find out how to find a $\mathbf{w}$ to minimize the error function.

### Solve The Euqation

$$
E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N[w_0 + w_1 x_n + \cdots + w_M x_n^M - t_n]^2\\
=\frac{1}{2}\sum_{n-1}^N \left(\sum_{j=0}^M x_n^j w_j - t_n\right)^2\\
$$

Then we solve the derivative equation $\frac{\partial E(\mathbf{w})}{\partial w_i} = 0$

$$
    \sum_{n=1}^N \left(\sum_{j=0}^M x_n^j w_j - t_n\right)x_n^i = 0
$$
$$
    \sum_{j=1}^M \left[\left(\sum_{n=1}^j x_n^{i+j}\right)w_j\right] = \sum_{n=1}^N x_n^i t_n
$$
There are $(i-1)$ more equations like this, and we need to solve them all to get $W$. So we use $A_{ij} = \sum_{n=1}^N x_n^{i+j}$ and $T_i = \sum_{n=1}^N x_n^i t_n$ to simplify the equation. The $i_{th}$ equation looks like this:
$$
\sum_{j=0}^M A_{ij} w_i = T_i
$$
Combine $i$ quations and we get a matirx equation:
$$
    \begin{bmatrix}
        A_{00} & \cdots & A_{0M} \\
        \vdots & & \vdots \\
        A_{M0} & \cdots & A_{MM}
    \end{bmatrix}
    \begin{bmatrix}
        w_0 \\
        \vdots \\
        w_M
    \end{bmatrix}
     = 
    \begin{bmatrix}
        x_1^0 & \cdots & x_n^0 \\
        \vdots & & \vdots \\
        x_1^M & \cdots & x_n^M
    \end{bmatrix}
    \begin{bmatrix}
        t_1 \\
        \vdots\\
        t_n
    \end{bmatrix}
$$
We notice that
$$
    \begin{bmatrix}
        A_{00} & \cdots & A_{0M} \\
        \vdots & & \vdots \\
        A_{M0} & \cdots & A_{MM}
    \end{bmatrix}
    = 
    X\cdot X^T
    \\
$$
$$
    X^T = 
    \begin{bmatrix}
        x_1^0 & \cdots & x_n^0 \\
        \vdots & & \vdots \\
        x_1^M & \cdots & x_n^M
    \end{bmatrix}
$$
So we could just solve this equation to get the $\mathbf{w}$:
$$
    X^T\cdot X \cdot \mathbf{w} = X^T \cdot \mathbf{t}
$$

It's exciting that so many equations end up being solved by just a single equation. Using python with package numpy we can easily solve this within a line:
```python
w = np.linalg.solve(X.T @ X, X.T @ t)
```

### Overfitting And Regularization
As we mentioned [above](#overfitting),the bigger the $M$, the more accurate the polynomial fit the training set. But if $M$ is too large then maybe the $y(x,\mathbf{w})$ will look like this:

![overfit](/images/overfit.png)

So we add a penalty term to the error function:
$$E(\mathbf{w}) = \frac{1}{2} \sum_{n = 1}^N[y(x_n,\mathbf{w}) - t_n]^2 + \frac{\lambda}{2}\|\mathbf{w}\|
$$
Now if $\mathbf{w}$ is too strange, $\| \mathbf{w} \|$ will become relatively large, which is not expected.

Similarly, solve this equation and we will get:
$$
    \left( X^T\cdot X + \lambda \cdot I \right)\cdot \mathbf{w} = X^T \cdot \mathbf{t}
$$

### Regularized Curve Fitting In Python

```python
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
np.random.seed(1234)

# Create training data and test data
def createSin(x):
    return np.sin(2 * np.pi * x)

def createTrainData(createSin, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = createSin(x) + np.random.normal(scale = std, size = x.shape)
    return x, t

x_train, y_train = createTrainData(createSin, 11, 0.25)
x_test = np.linspace(0, 1, 101)
y_test = createSin(x_test)

# compute X
def createX(x, degree):
    if x.ndim == 1:
        x = x[:, None]
    x_t = x.transpose()
    X = [np.ones(len(x))]
    for i in range(1,degree+1):
        for items in itertools.combinations_with_replacement(x_t, i):
            X.append(functools.reduce(lambda x, y: x * y, items ))
    return np.asarray(X).transpose()

degree = 9
X_train = createX(x_train, degree)
X_test = createX(x_test, degree)


# solve equation, return w
def solve(X, t, learnrate):
    return np.linalg.solve(learnrate * np.eye(np.size(X, 1)) + X.T @ X, X.T @ t)
    
# predict, return t
def predict(X, w):
    return X @ w

learnrate = 1e-3
w = solve(X_train, y_train, learnrate)
y = predict(X_test, w)

# plot
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M=9", xy=(-0.15, 1))
plt.show()
```
Code works well and it dosen't overfit when $M = 9$. 
![regularize](/images/regularize.png)
